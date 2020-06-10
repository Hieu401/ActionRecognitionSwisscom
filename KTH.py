''' test '''

import os
import re
import math  # for mathematical operations
# allows for unix style path finding
from glob import glob
import cv2  # for capturing videos

import matplotlib.pyplot as plt  # for plotting the images
import numpy as np  # for mathematical operations

# basically CSV
import pandas as pd

import tensorflow.keras

from skimage.transform import resize  # for resizing imagespylint --generate-rcfile > .pylintrc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# for preprocessing the images
import tensorflow.keras.preprocessing.image  as image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
import tensorflow.keras.utils
from tensorflow.keras.callbacks import ModelCheckpoint

from scipy import stats as s


# show progress bar, lol
from tqdm import tqdm

def PrepDataFrames():
    # creating a list of video name for test
    wavingTestVideoNames = os.listdir("./WavingClassifierDS/testdata/handwaving")
    nonWavingTestVideoNames = os.listdir("./WavingClassifierDS/testdata/non-waving")

    # creating a list of video names for training
    wavingTrainingVideoNames = os.listdir("./WavingClassifierDS/trainingdata/handwaving")
    nonWavingTrainingVideoNames = os.listdir("./WavingClassifierDS/trainingdata/non-waving")

    # creating a list of tag names for test
    wavingTestVideoTags = []
    nonWavingTestVideoTags = []

    # creating a list of tag names for training
    wavingTrainingVideoTags = []
    nonWavingTrainingVideoTags = []

    # adding the tags to the tags list
    for i in wavingTestVideoNames:
        wavingTestVideoTags.append("waving")

    for i in nonWavingTestVideoNames:
        nonWavingTestVideoTags.append("non-waving")

    for i in wavingTrainingVideoNames:
        wavingTrainingVideoTags.append("waving")

    for i in nonWavingTrainingVideoNames:
        nonWavingTrainingVideoTags.append("non-waving")

    wavingTestDict = {'video_name': wavingTestVideoNames, 'video_tag': wavingTestVideoTags}
    nonWavingTestDict = {'video_name': nonWavingTestVideoNames, 'video_tag': nonWavingTestVideoTags}
    wavingTrainingDict = {'video_name': wavingTrainingVideoNames, 'video_tag': wavingTrainingVideoTags}
    nonWavingTrainingDict = \
        {'video_name': nonWavingTrainingVideoNames, 'video_tag': nonWavingTrainingVideoTags}

    # creating a dataframe (tabel) having video names
    train = pd.DataFrame(data=wavingTrainingDict)
    tempTrain = pd.DataFrame(data=nonWavingTrainingDict)
    train = train.append(tempTrain)

    test = pd.DataFrame(data=wavingTestDict)
    tempTest = pd.DataFrame(data=nonWavingTestDict)
    test = test.append(tempTest)

    # select all rows except the last one
    train = train[:-1]
    # shuffle them
    train = train.sample(frac=1).reset_index(drop=True)
    # returns the first (n) row(s); default 1 row
    train.head()
    # select all rows except the last one
    test = train[:-1]
    # shuffle them
    test = test.sample(frac=1).reset_index(drop=True)
    # returns the first (n) row(s); default 1 row
    test.head()
    return train, test

def CreateFrames(train, folder):
    # storing the frames from training videos
    # for i to the amount of rows
    for i in tqdm(range(train.shape[0])):
        COUNT = 0
        # first video name
        videoFile = train.iloc[i, 0]
        videoLocation = os.path.join(R"D:/PycharmProjects/WavinFlag/WavingClassifierDS/trainingdata/all/")
        cap = cv2.VideoCapture(videoLocation + videoFile)  # capturing the video from the given path
        # get frame rate
        frameRate = cap.get(5)
        X = 1
        while cap.isOpened():
            # current frame number
            frameId = cap.get(1)
            # scroll through each frame, ret is true or false, false if there is no frame
            ret, frame = cap.read()
            if not ret:
                break
            if frameId % math.floor(frameRate) == 0:
                # storing the frames in a new folder named train_1
                filename = './WavingClassifierDS/' + folder + '_1/' + videoFile + "_frame" + str(COUNT) + ".jpg"
                COUNT += 1
                cv2.imwrite(filename, frame)
        cap.release()

def createCSV():
    # getting the names of all the images in an array
    images = glob("./WavingClassifierDS/train_1/*.jpg")
    train_image = []
    train_class = []
    for i in tqdm(range(len(images))):
        # creating the image name
        train_image.append(re.sub(R'./WavingClassifierDS/train_1\\', '', images[i]))
        # creating the class of image
        # if the term handwaving is in the name
        if re.search("handwaving", images[i]):
            train_class.append("waving")
        else:
            train_class.append("non-waving")

    # storing the images and their class in a dataframe
    train_data = pd.DataFrame()
    train_data['image'] = train_image
    train_data['class'] = train_class

    # converting the dataframe into csv file
    train_data.to_csv('./train_new.csv', header=True, index=False)

def CreateTrainingArray(train):
    train_image = []

    # for loop to read and store frames
    for i in tqdm(range(train.shape[0])):
        # loading the image and keeping the target size as (224,224,3)
        img = image.load_img('./WavingClassifierDS/train_1/' + train.iloc[i, 0], target_size=(224, 224, 3))
        # converting it to array
        img = image.img_to_array(img)
        # normalizing the pixel value
        img = img/255
        # appending the image to the train_image list
        train_image.append(img)

    # converting the list to a numpy array containing computer readable images
    X = np.array(train_image)

    # list of classes
    y = train['class']

    # creating the training and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)

    # creating dummies of target variable for train and validation set, basically turns categories into numbers so the neuro network can understand
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return X_train, X_test, y_train, y_test

def TrainModel(training_data, test_data, training_categories, test_categories):
    # creating the base model of pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)

    # extracting features for training frames
    training_data = base_model.predict(training_data)

    # extracting features for validation frames
    test_data = base_model.predict(test_data)

    # reshaping the training as well as validation frames in single dimension
    training_data = training_data.reshape(training_data.shape[0], 7*7*512)
    test_data = test_data.reshape(test_data.shape[0], 7*7*512)

    # normalizing the pixel values
    max = training_data.max()
    training_data = training_data/max
    test_data = test_data/max

    #defining the model architecture
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(training_data.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    #save weights of best model
    mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # training the model
    model.fit(training_data, training_categories, epochs=200, validation_data=(test_data, test_categories), callbacks=[mcp_save], batch_size=128)

def EvaluateModel(training_data, evaluation_DF):

    # list of classes
    y = evaluation_DF['video_tag']
    y = pd.get_dummies(y)

    base_model = VGG16(weights='imagenet', include_top=False)

    # extracting features for training frames
    training_data = base_model.predict(training_data)

    # reshaping the training as well as validation frames in single dimension
    training_data = training_data.reshape(training_data.shape[0], 7*7*512)

    # normalizing the pixel values
    max = training_data.max()
    training_data = training_data/max

    #defining the model architecture
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(training_data.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # loading the trained weights
    model.load_weights("weight.hdf5")

    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # # creating the tags
    # train = pd.read_csv('./train_new.csv')
    # y = train['class']
    # y = pd.get_dummies(y)

    # creating two lists to store predicted and actual tags
    predict = []
    actual = []

    # for loop to extract frames from each test video
    for i in tqdm(range(evaluation_DF.shape[0])):
        count = 0
        videoFile = evaluation_DF.iloc[i, 0]
        videoLocation = os.path.join(R"D:/PycharmProjects/WavinFlag/WavingClassifierDS/trainingdata/all/")
        cap = cv2.VideoCapture(videoLocation + videoFile)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        x=1

        # removing all other files from the temp folder
        files = glob('./temp/*')
        for f in files:
            os.remove(f)

        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if not ret:
                break
            if (frameId % math.floor(frameRate) == 0):
                # storing the frames of this particular video in temp folder
                filename = './temp/' + "_frame%d.jpg" % count
                count+=1
                cv2.imwrite(filename, frame)
        cap.release()
        
        # reading all the frames from temp folder
        images = glob("temp/*.jpg")
        
        prediction_images = []
        for i in range(len(images)):
            img = image.load_img(images[i], target_size=(224, 224, 3))
            img = image.img_to_array(img)
            img = img/255
            prediction_images.append(img)
            
        # converting all the frames for a test video into numpy array
        prediction_images = np.array(prediction_images)
        # extracting features using pre-trained model
        prediction_images = base_model.predict(prediction_images)
        # converting features in one dimensional array
        prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
        # predicting tags for each array
        prediction = model.predict_classes(prediction_images)
        print(prediction)
        # appending the mode of predictions in predict list to assign the tag to the video
        predicted_tag = y.columns.values[s.mode(prediction)[0][0]]
        predict.append(predicted_tag)
        # appending the actual tag of the video
        actual.append(evaluation_DF.iloc[i, 1])

        print(evaluation_DF.iloc[i, 0], evaluation_DF.iloc[i, 1], predicted_tag)

    # checking the accuracy of the predicted tags
    print(accuracy_score(predict, actual)*100)

trainDF, testDF = PrepDataFrames()

# CreateFrames(trainDF, "train")

# createCSV()

# CreateFrames(testDF, "test")

# dataframe
trainframe = pd.read_csv('./train_new.csv')
trainframe.head()

data1, data2, train_categories, testing_categories = CreateTrainingArray(trainframe)
# TrainModel(data1, data2, train_categories, testing_categories)
EvaluateModel(data2, trainDF)
