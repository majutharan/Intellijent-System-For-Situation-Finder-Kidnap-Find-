from typing import Any, Union

import requests
from django.db.models import QuerySet
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, request

# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from pydub import AudioSegment
from rest_framework.decorators import api_view
# from rest_framework.utils import json
from url import url

from kidnap.forms import VideoForm, AudioForm, RegisterForm, IdForm
from kidnap.models import Video, Audio, User
import json

#######################################################################
# first comment
#################################################
import numpy as np
import cv2
import os
import tkinter as Tk
import imp
import os
import cv2
import glob

import numpy as np
import cv2
import os

##################################################
import matplotlib.pyplot as plt
import numpy
from numpy import mean
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
import os
import time as t
# based on random forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

######################################################

import statistics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import json

from sklearn.externals import joblib
from .Send_Mail import mail

#############################################################

from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
###################################################################################


from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json


@csrf_exempt
def Root(request):
    text = """<h1>Welcome to kidnap finding API.</h1>"""
    return HttpResponse(text)


@csrf_exempt
def UploadVideo(request):
    try:
        saved = False

        if request.method == "POST":
            # Get the posted form
            videoForm = VideoForm(request.POST, request.FILES)

            if videoForm.is_valid():
                video = Video()
                video.uid = videoForm.cleaned_data["uid"]
                video.video = videoForm.cleaned_data["video"]
                video.save()
                saved = True
                FindKidnap(video.uid)

                return HttpResponse('Upload Image successful.')
        else:
            imageForm = VideoForm()

        status_code = 500
        message = "The request is not valid."
        explanation = "bad credentials"
        return JsonResponse({'message': message, 'explanation': explanation}, status=status_code)
    except RuntimeError as e:
        print("Thred error")
        return Response(e)
    except FileNotFoundError as fnf_error:
        print(fnf_error)

@csrf_exempt
def UploadAudio(request):
    saved = False

    if request.method == "POST":
        # Get the posted form
        audioForm = AudioForm(request.POST, request.FILES)

        if audioForm.is_valid():
            audio = Audio()
            audio.uid = audioForm.cleaned_data["uid"]
            audio.audio = audioForm.cleaned_data["audio"]
            audio.save()
            saved = True
            return HttpResponse('successfully saved!')
    else:
        audioForm = AudioForm()

    status_code = 500
    message = "The request is not valid."
    explanation = "bad credentials"
    return JsonResponse({'message': message, 'explanation': explanation}, status=status_code)


@csrf_exempt
def UserRegister(request):
    saved = False

    if request.method == "POST":
        # Get the posted form
        registerForm = RegisterForm(request.POST)

        if registerForm.is_valid():
            user = User()
            user.id = registerForm.cleaned_data["id"]
            user.name = registerForm.cleaned_data["name"]
            user.age = registerForm.cleaned_data["age"]
            user.place = registerForm.cleaned_data["place"]
            user.work = registerForm.cleaned_data["work"]
            user.sex = registerForm.cleaned_data["sex"]
            user.mail_id = registerForm.cleaned_data["mail_id"]
            user.save()
            saved = True
            return HttpResponse('User successfully registered!')
    else:
        registerForm = RegisterForm()

    status_code = 500
    message = "The request is not valid."
    explanation = "bad credentials"
    return JsonResponse({'message': message, 'explanation': explanation}, status=status_code)


# Testing purpose


@csrf_exempt
def GetVideo(request):
    if request.method == "POST":
        # Get the posted form
        idForm = IdForm(request.POST)
        if idForm.is_valid():
            user = User()
            user.id = idForm.cleaned_data["id"]
            # print('userId', user.id)
            # # FindKidnap(user.id)
            # # path = 'upload/videos/VID_20181113_112339.mp4'
            # audio = Audio.objects.filter(uid=user.id)
            # pathAudio = list(audio.values('audio')).pop(0)
            # pathAudio = "".join(pathAudio['audio'])
            # print(pathAudio)
            # inputVideo(path)

            userDetails = User.objects.filter(id=user.id)
            userList = list(userDetails.values('name', 'age', 'place', 'work', 'sex', 'mail_id'))
            users = userList.pop(0)
            print(users)
            user = User()
            user.name = users['name']
            user.age = users['age']
            user.place = users['place']
            user.work = users['work']
            user.sex = users['sex']
            user.mail_id = users['mail_id']

            print(user.name, user.age)

            return HttpResponse('User successfully registered!' + user.id)
        else:
            registerForm = RegisterForm()

    status_code = 500
    message = "The request is not valid."
    explanation = "bad credentials"
    return JsonResponse({'message': message, 'explanation': explanation}, status=status_code)


def inputVideo(path):
    path = path
    if path.lower()[-3:] == "mp4":
        extractAudio(path)


def extractAudio(path):
    global dst
    inputVideo = path
    path = path.split('/')
    path = path[2]

    dst = "kidnap/converter/audio_converter/wav_audios/" + path[:-3] + "wav"
    # file_name = path
    # file_name = file_name[:-3]
    # input_audio = dst
    cmd = "ffmpeg -i {} -vn  -ac 2 -ar 44100 -ab 320k -f wav {}".format(inputVideo, dst)
    os.popen(cmd)


# def wavConverter(file_name, input_audio):
#     input_audio = input_audio
#     dst = 'kidnap/converter/audio_converter/wav_audios/' + file_name + 'wav'
#     if input_audio.lower()[-3:] == "mp3":
#         sound = AudioSegment.from_mp3(input_audio)
#         sound.export(dst, format="wav")


def FindKidnap(uid):
    try:
        # uid = json.loads(uid.body)
        # id = str(uid)
        # print(id)
        # audio = Audio.objects.filter(uid='3')
        # pathVideo = audio.values('audio')
        #
        # audio = Audio.objects.filter(uid=id)
        # pathAudio = audio.values('audio')
        # print('image_path', pathVideo)
        # images_test = list(audio.values('uid', 'audio'))
        # images_test2 = images_test.pop(0)
        # print('list_path', images_test2['audio'])
        # str1 = "".join(images_test2['audio'])
        # userDetails = User.objects.filter(id = uid)

        # get user details

        userDetails = User.objects.filter(id=uid)
        userList = list(userDetails.values('name', 'age', 'place', 'work', 'sex', 'mail_id'))
        users = userList.pop(0)
        print(users)
        user = User()
        user.name = users['name']
        user.age = users['age']
        user.place = users['place']
        user.work = users['work']
        user.sex = users['sex']
        user.mail_id = users['mail_id']

        video = Video.objects.filter(uid=uid)
        pathVideo = list(video.values('video')).pop(0)
        pathVideo = "".join(pathVideo['video'])
        # print(pathVideo)
        inputVideo(pathVideo)
        # Video.objects.filter(id=video.last()).delete()
        print("bsdjkbfjsdbjbdb", dst)
        t.sleep(3)
        MyAudio = dst
        # MyAudio = "kidnap/converter/audio_converter/wav_audios/Screencast_2018-10-29_171124.wav"
        meanOfDecibel = []
        meanOfAplitude = []

        amplitudeData = "amplitudeDatamyvoice.csv"
        decibelData = "decibelDatamyvoice.csv"

        # for f in full_file_paths:
        #     if f.endswith(".wav"):

        myAudio = MyAudio
        print(myAudio)
        # Read file and get sampling frequency and sound object.
        samplingFreq, mySound = wavfile.read(myAudio)
        frequency, time, spectogram = signal.spectrogram(mySound, samplingFreq)

        # print sound object Ex: [1355, 955] (samples)
        print("my_sound:", mySound)

        # print sampling value(samples/sec)[Hz].
        print("sampling_freq:", samplingFreq)

        # Find the data type of sound file.
        # Ex: int 16, int 32
        mySoundDataType = mySound.dtype

        # print the value of sound bit.
        print('my_Sound_DataType:', mySoundDataType)

        print('Max value of sampling:', mySound.max())
        print('Min value of sampling:', mySound.min())
        print('mean of sound:', mean(mySound))

        # Convert the sound array to floating points [-1 to +1]
        mySound = mySound / 2 ** 15

        # Print the floating points.
        print('sound value in float:', mySound)

        # Check sound points. Its dual-channel or mono channel
        mySoundShape = mySound.shape

        # Analysis the how many samples are in sound object.
        samplePoints = float(mySound.shape[0])

        # Print the samples count.
        print('SamplePoints', samplePoints)
        print('sound shape', mySoundShape)

        # Find the signal duration = samples / (samples / time)
        signalDuration = mySound.shape[0] / samplingFreq

        # Print the duration of wave file.
        print('signal duration', signalDuration)

        # If two channel then get one channel

        print('mySound', mySound)

        mySoundOneChannel = mySound[:, 0]
        print('mySoundOneChannel:', mySoundOneChannel)

        # Arange the value of sample points within 1 to 169629 [space 1]
        timeArray = numpy.arange(0, samplePoints, 1)
        print('time array:', timeArray)

        # time array = samplePints array / (samples / t) => t
        timeArray = timeArray / samplingFreq
        print('time array new:', timeArray)

        # Change to s(seconds)  to ms(milli seconds)
        timeArray = timeArray * 1000

        # Print s(seconds) value to ms(milli seconds)
        print('time array * 1000 :', timeArray)
        meanOfAplitude.append(float(mean(mySoundOneChannel)) * (-10000000000))
        print('mean of amplitude', meanOfAplitude)
        numpy.savetxt(amplitudeData, meanOfAplitude, fmt="%s")

        # make graph to amplitude vs time
        # plt.plot(timeArray, mySoundOneChannel, color='G')
        # plt.ylabel('Amplitude')
        # plt.xlabel('Time')
        # plt.show()

        # calculate the sound length
        mySoundLength = len(mySound)
        print('my sound length is :', mySoundLength)

        # fast frequency transformation of sound clip
        # Analysis the sound clip via frequency.
        fftArray = fft(mySoundOneChannel)
        print('fast frequency transformation of mySoundOneChannel)', fftArray)

        numUniquePoints = int(numpy.ceil((mySoundLength + 1) // 2))
        print('numUniquePoints:', numUniquePoints)

        fftArray = fftArray[0:numUniquePoints]
        print('fastFrequencyTransformation:', fftArray)

        fftArray = abs(fftArray)
        print('fftArray:', fftArray)

        fftArray = fftArray / float(mySoundLength)
        print('fftArray:', fftArray)

        # change to positive numbers.
        fftArray = fftArray ** 2

        # Multiply by two
        # Odd NFFT excludes Nyquist point
        if mySoundLength % 2 > 0:  # we've got odd number of points in fft
            fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2

        else:  # We've got even number of points in fft
            fftArray[1:len(fftArray) - 1] = fftArray[1:len(fftArray) - 1] * 2

        freqArray = numpy.arange(0, numUniquePoints, 1.0) * (samplingFreq / mySoundLength)

        # log algorithm

        log = 10 * numpy.log10(fftArray)
        print('log', log)
        print('mean', mean(log))

        # Plot the frequency
        # plt.plot(freqArray / 1000, 10 * numpy.log10(fftArray), color='B')
        # plt.xlabel('Frequency (Khz)')
        # plt.ylabel('Power (dB)')
        # plt.show()

        # Get List of element in frequency array
        # print freqArray.dtype.type
        freqArrayLength = len(freqArray)
        print("freqArrayLength =", freqArrayLength)
        meanOfDecibel.append(mean(log) * (-1))
        numpy.savetxt(decibelData, meanOfDecibel, fmt="%s")

        # Print FFtarray information
        print("fftArray length =", len(fftArray))

        spectogram1D = spectogram[:, 0]
        spectogram1D = spectogram1D[:, 0]

        print('frequency is :', frequency)
        print('time is :', time)
        print('spectogram is :', spectogram1D)

        print('meanOfAplitude', meanOfAplitude)
        print('meanOfDecibel', meanOfDecibel)

        ############################

        data = pd.read_csv("kidnap/fullaplitudedataset.csv")
        # print(data.head())

        X = data[['A', 'D']]  # Features
        y = data['emotion']  # Labels

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 70% training and 30% test

        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=25)

        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print(X_train)

        sns.heatmap(data.corr())
        # plt.show()

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        meanOfAplitude = meanOfAplitude[0]
        meanOfDecibel = meanOfDecibel[0]

        print('meanOfAplitude', meanOfAplitude)
        print('meanOfDecibel', meanOfDecibel)

        # predict sistutation
        # print(clf.predict([[meanOfAplitude, meanOfDecibel]]))
        emotion = clf.predict([[meanOfAplitude, meanOfDecibel]])
        final_emotion = emotion[0]
        print(final_emotion)

        ########################################################################################
        # try:
        #     if not os.path.exists('/home/majutharan/Documents/projects/Final_research/KidnapFind/VideoFrames'):
        #         os.makedirs('/home/majutharan/Documents/projects/Final_research/KidnapFind/VideoFrames')
        #     else:
        #         frameImgs = glob.glob(os.path.join('/home/majutharan/Documents/projects/Final_research/KidnapFind/VideoFrames', '*g'))
        #         for fi in frameImgs:
        #             os.remove(fi)
        # except OSError:
        #     print("Error in create file")

        cap = cv2.VideoCapture(pathVideo)
        print(pathVideo)
        print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")

        currentFrame = 0

        sec = 0
        frameRate = 0.7
        success = True

        def getFrame(sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = cap.read()

            if hasFrames:
                name = 'VideoFrames/frame' + str(currentFrame) + '.jpg'
                print('Creating...' + name)
                cv2.imwrite(name, image)
            return hasFrames

        while success:
            success = getFrame(sec)
            sec = sec + frameRate
            sec = round(sec, 2)
            currentFrame = currentFrame + 1
            if success == False:
                break
        # #####################################################################################
        face_cascade = cv2.CascadeClassifier(
            '/home/majutharan/Documents/projects/Final_research/KidnapFind/kidnap/cascade/haarcascade_frontalcatface.xml')
        eye_cascade = cv2.CascadeClassifier('/home/majutharan/Documents/projects/Final_research/KidnapFind/kidnap/cascade/haarcascade_eye.xml')

        people_count = []

        def get_filepaths(directory):
            file_paths = []

            for root, directories, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)

            return file_paths

        full_file_paths = get_filepaths("/home/majutharan/Documents/projects/Final_research/KidnapFind/VideoFrames")

        for f in full_file_paths:
            if f.endswith(".jpg"):
                myimage = f
                img = cv2.imread(myimage)
                grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(grayImage, 1.9)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    roi_gray = grayImage[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.5)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), 1)
                print(len(faces))

                people_count.append(len(faces))

                cv2.imshow('img', img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                cv2.destroyAllWindows()

        print(people_count)
        total_people_count = max(people_count)
        print(total_people_count)

        ########################################################################################

        filename = 'kidnap/finalized_model.sav'
        # load the model from disk
        loaded_model = joblib.load(filename)
        # result = loaded_model.score(X_test, Y_test)
        age = int(user.age)
        place = user.place
        work = user.work
        sex = user.sex
        if place == "jaffna":
            placeId = 1
        if place == "colombo":
            placeId = 2
        if place == "kandy":
            placeId = 3
        if work == "working":
            workId = 1
        if work == "notworking":
            workId = 2
        if sex == "male":
            genderId = 1
        if sex == "female":
            genderId = 2

        print(placeId, workId, genderId)
        # decission = loaded_model.predict([[25, 0, 0, 1, 23, 1, 1]])
        decission = loaded_model.predict([[total_people_count, 0, 0, placeId, age, workId, genderId]])
        print(decission)
        emotion = final_emotion
        email = []
        email = user.mail_id

        if (decission == "y"):
            mail(emotion, user.name, email)
            HttpResponse("There is kidnap \n" + "Name: " +user.name + "\n" + "Age: " + user.age + "\n" + "place: " + user.place)

        ########################################################################
        return HttpResponse("success")

    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
    except RuntimeError as e:
        print("Thred error")
        return Response(e)
    except FileNotFoundError as fnf_error:
        print(fnf_error)


def Login(request):
    try:
        return HttpResponse('sucessfully registered')
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
