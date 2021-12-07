import time
import os
import sys
from os import path
from os.path import join
import glob
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
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import getSamples as gs
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

 
####################CSV WRITING STARTS#############################
###Creating train data
metadata_file = './accent_data.csv'

#Check if the file exists that can be loaded to the dataFrame 
df = pd.read_csv(metadata_file)
df['relative_path'] =  df['Speaker'].astype(str) 
df = df[['relative_path','Speaker','Gender','Native_Language']]
#print(df)

#Helper method to extract the MFCC features of a WAV file
def extract_features(file_name):
    #Load the audio file
    audio,sample_rate = librosa.load(file_name,res_type = 'kaiser_fast')
    #Obtain the MFCC array for each such word utterance
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate)
    #Normalize the MFCC array by rows
    mfcc_normalized = mfcc/np.linalg.norm(mfcc, ord=2, axis=1, keepdims=True)
    #Reshape the MFCC array to get a 1D-array
    mfcc_reshaped = np.reshape(mfcc_normalized, -1)
    return mfcc_reshaped

if __name__ == "__main__":
    if(len(sys.argv)!=4):
        raise Exception("Incorrect number of arguments! Usage: accent_recognition_suitcase.py --load <training file> <testing file> or --create  <training file> <testing file>")
    else:
        create_file = sys.argv[1]
        train_file = sys.argv[2]
        test_file = sys.argv[3]

        if(create_file == "--create"):
            #Create a test CSV file of the entire dataset with all the features
            print("===============================Extracting features===========================\n")      
            for i in range(df.shape[0]):
                matches = []
                matches_noise = []
                features = []
                gender = []
                language = []
                speaker = []
                speaker_name = df.loc[i].at['Speaker']

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
                pathOutput = './suitcase_corpus/padded_spliced_audio/'
                pathOutput_noised = './suitcase_corpus/padded_noised_spliced_audio/'
                #print(pathAudio)
                print(pathOutput)
                matches = [os.path.abspath(pathOutput + '/' + fname) for fname in os.listdir(pathOutput) if fname.startswith(df.loc[i].at['Speaker'])]
                matches_noise = [os.path.abspath(pathOutput_noised + '/' + fname) for fname in os.listdir(pathOutput) if fname.startswith(df.loc[i].at['Speaker'])]
                matches.extend(matches_noise)

                for match in matches: 
                    print("file is",match)
                    data = extract_features(match)
                    features.append(data)
                    gender.append(a)
                    language.append(b)
                    speaker.append(speaker_name)

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
                df_csv.to_csv(test_file, index=False, header=False, mode = 'a')
                print("\n----------Successfully Added Information for: ", speaker_name, " to the Testing File-----------\n") 


        #Load the training and test datasets
        if(os.path.isfile(train_file)==False):
            raise Exception("No such training data available: ", train_file)
        if(os.path.isfile(test_file)==False):
            raise Exception("No such testing data available: ", test_file)

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
        time_start = time.time()
        gnb = GaussianNB()
        print("Shapes for the training dataset are: ", X_train.shape, Y_train.shape, "\n")
        gnb.fit(X_train, Y_train)
        Y_pred = gnb.predict(X_test)
        print('Gaussian Naive Bayes Regression for all speakers done! Time elapsed: {} seconds'.format(time.time()-time_start))
        print('Train/Test split results:')
        print(gnb.__class__.__name__+" accuracy is %2.3f" % accuracy_score(Y_test, Y_pred))
        print(gnb.__class__.__name__+" F1 score is %2.3f" % f1_score(Y_test, Y_pred,average='macro'))
        print(gnb.__class__.__name__+" Recall score  is %2.3f" % recall_score(Y_test, Y_pred,average='macro'))
        print(gnb.__class__.__name__+" Precision score  is %2.3f" % precision_score(Y_test, Y_pred,average='macro'))
        titles_options = [
        ("Gaussian Naive Bayes (Arabic=1, Hindi=2, Chinese=3)", "true")
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
        #plt.show()
        plt.savefig("GaussianNB_Suitcase_Conf.png")
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
        ("Logistic Regression (Arabic=1, Hindi=2, Chinese=3)", "true")
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
        #plt.show()
        plt.savefig("Logistic_Suitcase_Conf.png")
        print(classification_report(Y_test, Y_pred))
            