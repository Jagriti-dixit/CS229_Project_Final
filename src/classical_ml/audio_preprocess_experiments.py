##########################################################
#######File to add noise to the data and pad it###########
##########################################################

import os
from os import path
from os.path import join

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
from audio_preprocessing_helpers import AudioAugmentation


#### Import Comet for experiment tracking and visual tools
####
#iteration = "noise"
#iteration = "pad"
#actual dataset reading
#download_path = Path.cwd()/'l2arctic_release_v5.0.zip'
metadata_file = './accent_data.csv'

#Check if the file exists that can be loaded to the dataFrame 

df = pd.read_csv(metadata_file)
# print(df.head(n=1))

speaker = []
# duration =[]
mfcc_words = []
# wav_name = []
aa = AudioAugmentation()

#Padding all the split word utterances obtained and adding noise to them 
for i in range(df.shape[0]):
    # if df.loc[i].at['Speaker'] == "HKK":
    speaker = df.loc[i].at['Speaker']
    pathAudio = './' + df.loc[i].at['Speaker'] + '/spliced_audio/'
    paddedOutput = './' + df.loc[i].at['Speaker'] + '/padded_spl_audio/'
    noiseOutput = './' + df.loc[i].at['Speaker'] + '/noised_padded_spl_audio/'

    if not os.path.exists(noiseOutput):
        os.mkdir(noiseOutput)
    if not os.path.exists(paddedOutput):
        os.mkdir(paddedOutput)

    print("=============Padding & Adding Noise Beginning for: ", speaker, " =================\n")
    audio_files = librosa.util.find_files(pathAudio, ext=['wav']) 
    files = np.asarray(audio_files)
    print("Succesfully loaded all audio files for: ", speaker, "\n")
    for y in files:
        audio,sample_rate = librosa.load (y,res_type = 'kaiser_fast')
        data = aa.read_audio_file(y)
        #Pad the spliced word utterance
        #The maximum length of a spliced word utterance across all our speakers is 2.35 seconds, padding according to this value
        max_value = 2.35
        data_pad = aa.pad(data, max_value)
        y_pad = y.replace("/spliced_audio/","/padded_spl_audio/")
        aa.write_audio_file(y_pad, data_pad)

        #Add noise to spliced word utterances
        data_noise = aa.add_noise(data)
        #Pad the noised-spliced word utterances
        data_noise_pad = aa.pad(data_noise, max_value)
        y_noise = y.replace("/spliced_audio/","/noised_padded_spl_audio/")
        aa.write_audio_file(y_noise, data_noise_pad)
    
    print("=============Padding & Adding Noise Completed for: ", speaker, " =================\n")



