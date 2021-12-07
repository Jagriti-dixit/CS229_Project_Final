import os
from os import path
from os.path import join
import sys

from comet_ml import Experiment
import pydub
import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import math,random
import zipfile as zf
import soundfile as sf
import io
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm
import audio_preprocessing_helpers as ap
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score


#actual dataset reading
metadata_file = './accent_data.csv'
dirPath = "/home/paavni/processed_data/"
#Check if the file exists that can be loaded to the dataFrame 
df = pd.read_csv(metadata_file)
df['relative_path'] =  df['Speaker'].astype(str) 
df = df[['relative_path','Speaker','Gender','Native_Language','Data_Type']]
# print(df)

if(len(sys.argv)!=4):
    raise Exception("Incorrect number of arguments! Usage: accent_recognition.py --load <training file> <testing file> or --create <training file> <testing file>")
else:
    create_file = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]

    if(create_file == "--create"):
        #Create a training CSV file of the entire dataset with all the features
        print("===============================Extracting features===========================\n")
        #Assigning values for classification to gender and language
        for i in range(df.shape[0]):
        #for i in range(1):
            features = []
            gender = []
            language = []
            speaker = []
            speaker_name = df.loc[i].at['Speaker']
            data_type = df.loc[i].at['Data_Type']

            print("\n----------Adding Information for: ", speaker_name, " ---------------\n")
            if df.loc[i].at['Gender'] == "M":
                a = 1
            else:
                a = 0
            if df.loc[i].at['Native_Language'] == "Arabic":
                b = 1
            elif df.loc[i].at['Native_Language'] == "Hindi":
                b = 2
            elif df.loc[i].at['Native_Language'] == "Chinese":
                b = 3    

            #audioPath = './' + speaker_name + '/padded_spl_audio/'
            #audioPath_noise = './' + speaker_name + '/noised_padded_spl_audio/'
            audioPath = dirPath + speaker_name + '/padded_spliced_audio/'
            audioPath_noise = dirPath + speaker_name + '/padded_noised_spliced_audio/'
            #audio_files = librosa.util.find_files(audioPath, ext=['wav']) 
            audio_noise_files = librosa.util.find_files(audioPath_noise, ext=['wav']) 
            #audio_files.extend(audio_noise_files)
            # print(len(audio_files))
            files = np.asarray(audio_noise_files)
            for file_name in files: 
                # print("file is", file)
                data = ap.extract_features(file_name)
                features.append(data)
                gender.append(a)
                language.append(b)
                speaker.append(speaker_name)
                #break

            # Creating the pandas Dataframe for the training models
            features = np.asarray(features)
            gender = np.asarray(gender)
            language = np.asarray(language)
            speaker = np.asarray(speaker)
            print("\nFeature Matrix Shape for ", speaker_name, " is: ", features.shape, "\n")
            
            # Write this data to the CSV file (either test or train csv file according to the speaker's data_type in metadata file)
            df_csv = pd.DataFrame(data=features)
            df_csv[len(df_csv.columns)] = gender
            df_csv[len(df_csv.columns)] = language
            df_csv[len(df_csv.columns)] = speaker
            
            
            if(data_type == "Train"):
                df_csv.to_csv(train_file, index=False, header=False, mode = 'a')
                print("\n--------Successfully Added Information for: ", speaker_name, " to the Training File-----------\n")
            else:
                df_csv.to_csv(test_file, index=False, header=False, mode = 'a')
                print("\n----------Successfully Added Information for: ", speaker_name, " to the Testing File-----------\n")         
 
    
    
    #Load the training and test datasets
    if(os.path.isfile(train_file)==False):
        raise Exception("No such training data available: ", train_file)
    if(os.path.isfile(test_file)==False):
        raise Exception("No such testing data available: ", test_file)

    # train_data = pd.read_csv(train_file, header=None)
    # test_data = pd.read_csv(test_file, header=None)

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
   
    #Train the model
    X_train = train_data.iloc[:, :2040]
    Y_train = train_data.iloc[:, 2041]
    #print(Y_train)
    X_test = test_data.iloc[:, :2040]
    Y_test = test_data.iloc[:, 2041]
    #print("shapes of X_train y_train are",X_train.shape,y_train.shape)

    #Implement & Run the Gaussian Naive Bayes model
    gnb = GaussianNB()
    print("Shapes for the training dataset are: ", X_train.shape, Y_train.shape, "\n")

    gnb.fit(X_train, Y_train)
    Y_pred = gnb.predict(X_test)
    print('Train/Test split results:')
    print(gnb.__class__.__name__+" accuracy is %2.3f" % accuracy_score(Y_test, Y_pred))
    print(gnb.__class__.__name__+" F1 score is %2.3f" % f1_score(Y_test, Y_pred,average='macro'))
    print(gnb.__class__.__name__+" Recall score  is %2.3f" % recall_score(Y_test, Y_pred,average='macro'))
    print(gnb.__class__.__name__+" Precision score  is %2.3f" % precision_score(Y_test, Y_pred,average='macro'))
    titles_options = [
    ("Gaussian Naive Bayes", "true")
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            gnb,
            X_test,
            Y_test,
            display_labels=gnb.classes_,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.show()
    print(classification_report(Y_test, Y_pred))

    #Implement & Run the Logistic Regression model
    logreg = LogisticRegression(max_iter = 10000,penalty = 'l2')

    # #X_train,y_train = make_classification(n_samples = 1000)
    print("Shapes for the training dataset are: ", X_train.shape, Y_train.shape, "\n")

    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    print('Train/Test split results:')
    print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(Y_test, Y_pred))
    print(logreg.__class__.__name__+" F1 score is %2.3f" % f1_score(Y_test, Y_pred,average='macro'))
    print(logreg.__class__.__name__+" Recall score  is %2.3f" % recall_score(Y_test, Y_pred,average='macro'))
    print(logreg.__class__.__name__+" Precision score  is %2.3f" % precision_score(Y_test, Y_pred,average='macro'))
    titles_options = [
    ("Logistic Regression", "true")
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            logreg,
            X_test,
            Y_test,
            display_labels=logreg.classes_,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.show()
    print(classification_report(Y_test, Y_pred))