# All code in this project was ran on the Google Colaboratory platform

# # Libraries and connections
# !pip install tensorflow
# !pip install dlib
# !pip install opencv-python

# imports
import glob
import cv2 as cv
import sys
from google.colab.patches import cv2_imshow
import os
import dlib
import keras
import tensorflow as tf
import pickle
import numpy as np
import h5py
import keras
from sklearn.utils import shuffle
from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D
from keras.layers import TimeDistributed, GRU, Dense, Dropout, Bidirectional, LSTM
import shutil
import re
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import \
    ImageDataGenerator, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


def extract_mouth(video_path):
    predictor_path = "/content/drive/MyDrive/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor(predictor_path)
    frames = []
    vc = cv.VideoCapture(video_path)

    PAD_PERCENT_OF_FRAME = 0.05

    pad_width = int(vc.get(cv.CAP_PROP_FRAME_WIDTH) * PAD_PERCENT_OF_FRAME)
    pad_height = int(vc.get(cv.CAP_PROP_FRAME_WIDTH) * PAD_PERCENT_OF_FRAME)

    # Read all the frames in the video and construct an array of grayscale mouth crop
    if vc.isOpened():
        while 1:
            ok, frame = vc.read()
            if not ok:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = detector(gray, 1)
            shape = predictor(gray, rects[0])
            xmouthpoints = [shape.part(x).x for x in range(48, 67)]
            ymouthpoints = [shape.part(x).y for x in range(48, 67)]
            maxx = max(xmouthpoints)
            minx = min(xmouthpoints)
            maxy = max(ymouthpoints)
            miny = min(ymouthpoints)
            # to show the mouth properly pad both sides
            crop_image = gray[miny - pad_height:maxy + pad_height, minx - pad_width:maxx + pad_width]
            crop_image = cv.resize(crop_image, (64, 64))
            frames.append(crop_image)

    vc.release()


# general variables

ERR_LOG_PATH = "/content/drive/MyDrive/logs.txt"
dataset_path = "/content/drive/MyDrive/data/"
lab_path = "/content/drive/MyDrive/data/video_lab/"
wild_path = "/content/drive/MyDrive/data/video_wild/"
checkpoints_path = '/content/drive/MyDrive/data/checkpoints/'
crop_width, crop_height = 64, 64

SIZE = (crop_width, crop_height)
CHANNELS = 1
NBFRAME = 29
BS = 32
num_epochs = 100
glob_pattern = '/content/drive/MyDrive/data/video_wild/train/{classname}/*.avi'


def get_list_of_all_words(ds_path) -> np.array:
    cuvinte = set()
    for root, dirs, files in os.walk(ds_path):

        if root.count(os.sep) == 7:
            cuvinte.add(root[root.rfind(os.sep) + 1:])
            dirs = []

    return np.array(sorted(list(cuvinte)))


cuvinte = get_list_of_all_words(wild_path)
classes = list(cuvinte)
print(cuvinte)

# Generator
class VideoFrameGenerator(Sequence):
    def __init__(
            self,
            glob_pattern: str,
            rescale=1 / 255.,
            nb_frames: int = 29,
            classes: list = None,
            batch_size: int = 16,
            target_shape: tuple = (64, 64),
            shuffle: bool = True,
            transformation: ImageDataGenerator = None,
            nb_channel: int = 3):

        self.glob_pattern = glob_pattern
        classes.sort()

        self.rescale = rescale
        self.classes = classes
        self.batch_size = batch_size
        self.nbframe = nb_frames
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation

        self._random_trans = []
        self.files = []

        for cls in classes:
            self.files += glob.glob(glob_pattern.format(classname=cls))

        self.files_count = len(self.files)
        self.indexes = np.arange(self.files_count)
        self.classes_count = len(classes)

        self.on_epoch_end()

        self._current = 0
        self._framecounters = {}
        print("Total data: %d classes for %d files" % (
            self.classes_count,
            self.files_count))

    def next(self):
        elem = self[self._current]
        self._current += 1
        if self._current == len(self):
            self._current = 0
            self.on_epoch_end()

        return elem

    def on_epoch_end(self):
        # executed by keras every time an epoch ends
        if self.transformation is not None:
            self._random_trans = []
            for _ in range(self.files_count):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return int(np.floor(self.files_count / self.batch_size))

    def __getitem__(self, index):
        # returns a collection of input data with the size of the batch size
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        transformation = None

        for i in indexes:
            if self.transformation is not None:
                transformation = self._random_trans[i]

            video = self.files[i]
            classname = self._get_classname(video)

            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.

            frames = self._get_frames(
                video,
                nbframe,
                shape)
            if frames is None:
                continue

            if transformation is not None:
                frames = [self.transformation.apply_transform(
                    frame, transformation) for frame in frames]

            images.append(frames)
            labels.append(label)

        return np.array(images), np.array(labels)

    def _get_classname(self, video: str) -> str:

        video = os.path.realpath(video)
        pattern = os.path.realpath(self.glob_pattern)

        pattern = re.escape(pattern)

        pattern = pattern.replace('\\*', '.*')
        pattern = pattern.replace('\\{classname\\}', '(.*?)')

        classname = re.findall(pattern, video)[0]
        return classname

    def _get_frames(self, video, nbframe):
        # reading of a video file
        cap = cv.VideoCapture(video)
        total_frames = 29
        orig_total = total_frames
        if total_frames % 2 != 0:
            total_frames += 1
        frames = []

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break
            frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            frame = img_to_array(frame) * self.rescale

            # keep frame
            frames.append(frame)

            if len(frames) == nbframe:
                break

        cap.release()

        return np.array(frames)


data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)

# initializing a generator for training data
train = VideoFrameGenerator(
    classes=classes,
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    shuffle=True,
    batch_size=BS,
    transformation=data_aug,
    target_shape=SIZE,
    nb_channel=CHANNELS)

# model

def build_convnet(shape=(64, 64, 3)):
    # four convolutions
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=shape,
                     padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(MaxPool2D())

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    # transform the output of previous layer to a 1 dimensional one (could be replaced with Flatten())
    model.add(GlobalMaxPool2D())
    return model


def action_model(shape=(29, 64, 64, 3), nbout=48):
    # get the convolutional layer
    convnet = build_convnet(shape[1:])

    # then create our final model
    model = keras.Sequential()
    # add the convnet with (29, 64, 64, 1) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    # a GRU or an LSTM could be used
    model.add(Bidirectional(GRU(64, return_sequences=True)))
    model.add(Bidirectional(GRU(64)))
    # Dense layers, for the actual decision making
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model


INSHAPE = (NBFRAME,) + (crop_width, crop_height) + (CHANNELS,)  # (29, 64, 64, 1)
model = action_model(INSHAPE, len(cuvinte))
optimizer = keras.optimizers.Adam(0.0001)
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

model.summary()


# training
# called by keras on epoch end (if conditions are met)
callbacks = [
    keras.callbacks.ModelCheckpoint(
         checkpoints_path + 'weights_lab_new.-{acc:.2f}.hdf5',
        monitor='acc', verbose=1, save_best_only=True, mode='auto'),
]

# main training loop
history = model.fit(
    train,
    epochs=num_epochs,
    callbacks=callbacks,
    batch_size=BS,
    verbose=1,
    steps_per_epoch=train.files_count/BS
)

# save accuracy over the steps
os.makedirs('/content/drive/MyDrive/data/histories', exist_ok=True)
with open('/content/drive/MyDrive/data/histories/wild_history_acc.dat', 'wb') as f:
        pickle.dump(history.history['acc'], f)
with open('/content/drive/MyDrive/data/histories/wild_history_loss.dat', 'wb') as f:
    pickle.dump(history.history['loss'], f)

# load metrics from saved files

with open('/content/drive/MyDrive/data/histories/wild_history_acc.dat', 'rb') as f:
    wild_acc = pickle.load(f)

with open('/content/drive/MyDrive/data/histories/wild_history_loss.dat', 'rb') as f:
    wild_loss = pickle.load(f)

# testing

# load weights from a file, so we won't have to train again
model.load_weights('/content/drive/MyDrive/data/checkpoints/weights_wild_new.-0.80.hdf5')

# create a generator for the testing data
test = VideoFrameGenerator(
    classes=classes,
    glob_pattern=glob_pattern.replace('train', 'test'),
    nb_frames=NBFRAME,
    shuffle=False,
    batch_size=121,
    target_shape=SIZE,
    nb_channel=CHANNELS)

# we get the predictions of the model for the given input
results = model.evaluate(test,
                         steps=test.files_count
                         )
# plooting of wild/lab accuracy increase/loss decrease over training time

plt.plot(range(len(wild_acc)), wild_acc, label="Accuracy for wild dataset")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Wild training over the epochs')
plt.legend()
plt.show()

plt.plot(range(len(wild_loss)), wild_loss, label="Loss for wild dataset", color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Wild training over the epochs')
plt.legend()
plt.show()

plt.plot(range(len(lab_acc)), lab_acc, label="Accuracy for lab dataset")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Lab training over the epochs')
plt.legend()
plt.show()

plt.plot(range(len(lab_loss)), lab_loss, label="Loss for lab dataset", color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Lab training over the epochs')
plt.legend()
plt.show()


# CER and WER calculations
print(results.shape)
correct_count = 0
wrong_characters = 0
characters_count = 0
test_labels = np.array(test[0][1])
for i in range(len(results)):
    predicted_idx = np.argmax(results[i])
    actual_idx = np.argmax(test_labels[i])
    predicted_word = cuvinte[predicted_idx]
    actual_word = cuvinte[actual_idx]

    if predicted_idx == actual_idx:
        correct_count += 1
        characters_count += len(predicted_word)
    else:
        # compute CER
        characters_count += len(predicted_word)
        if len(actual_word) > len(predicted_word):
            actual_word, predicted_word = predicted_word, actual_word

        for j in range(len(predicted_word)):
            if j < len(actual_word):
                if actual_word[j] != predicted_word[j]:
                    wrong_characters += 1
            else:
                wrong_characters += 1

images_cnt = len(results)

WER_wild = 1 - (correct_count / images_cnt)
CER_wild = (wrong_characters / characters_count)
print(f"WER = {WER_wild} \nCER = {CER_wild}")

# creating and plotting of the Confusion Matrix
l = [np.argmax(elem) for elem in np.array(test_labels)]
r = [np.argmax(elem) for elem in np.rint(np.array(results))]

confusion_m = confusion_matrix(l, r)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_m, display_labels=classes)
fig, ax = plot_confusion_matrix(
    conf_mat=confusion_m,
    show_absolute=False,
    show_normed=True,
    colorbar=True,
    figsize=(30, 30),
    class_names=classes)
plt.savefig('/content/drive/MyDrive/data/confusion/confusion_lab.png')

