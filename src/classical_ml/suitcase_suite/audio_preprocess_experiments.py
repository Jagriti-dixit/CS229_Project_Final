import os
from os import path
from os.path import join
import pydub
import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import math,random
import soundfile as sf
import getSamples as gs
from data_processing_helpers import AudioAugmentation
import sys


#iteration = "noise"
#iteration = "pad"
#actual dataset reading
download_path = Path.cwd()/'l2arctic_release_v5.0.zip'
metadata_file = './accent_data.csv'

if __name__ == "__main__":
    #Check if the file exists that can be loaded to the dataFrame 
    iteration = sys.argv[1]
    df = pd.read_csv(metadata_file)
    speaker = []
    duration =[]
    mfcc_words = []
    wav_name = []

    aa = AudioAugmentation()
    for i in range(df.shape[0]):
        pathAudio = './'  + 'suitcase_corpus'+ '/spliced_audio/'
        noiseOutput = './'  + 'suitcase_corpus'+ '/noised_spliced_audio/'

        paddedOutput = './' + 'suitcase_corpus'+ '/padded_spliced_audio/'
        paddednoisedOutput = './'  + 'suitcase_corpus'+ '/padded_noised_spliced_audio/'

        if not os.path.exists(noiseOutput):
            os.mkdir(noiseOutput)
        if not os.path.exists(paddedOutput):
            os.mkdir(paddedOutput)
        if not os.path.exists(paddednoisedOutput):
            os.mkdir(paddednoisedOutput)

        audio_files = librosa.util.find_files(pathAudio, ext=['wav']) 
        noise_audio_files = librosa.util.find_files(noiseOutput,ext =['wav'])
        print(len(audio_files))
        
        files = np.asarray(audio_files)
        noise_files = np.asarray(noise_audio_files)

        #Adds noise to the spliced words for data augmentation
        if iteration == "noise":
            for y in files:
                audio,sample_rate = librosa.load (y,res_type = 'kaiser_fast')
                t = librosa.get_duration(y=audio,sr=sample_rate)
                duration.append(t)
                wav_name.append(y)
                data = aa.read_audio_file(y)
                print("\nlength is ",len(data))
                data_noise = aa.add_noise(data)
                y_noise = y.replace("/spliced_audio/","/noised_spliced_audio/")
                aa.write_audio_file(y_noise,data_noise)

        #Pads the spliced words and the noise-added spliced words   
        if iteration == "pad":    
            for y in files:
            #As the maximum length of a word audio file is 2.35 seconds
                max_value = 2.35
                audio,sample_rate = librosa.load (y,res_type = 'kaiser_fast')
                data = aa.read_audio_file(y)
                data_pad = aa.pad(data,max_value)
                y_pad = y.replace("/spliced_audio/","/padded_spliced_audio/")
                aa.write_audio_file(y_pad,data_pad)

            for y in noise_files:
                max_value =2.35
                audio,sample_rate = librosa.load (y,res_type = 'kaiser_fast')
                data = aa.read_audio_file(y)
                data_pad = aa.pad(data,max_value)
                y_noise_pad = y.replace("/noised_spliced_audio/","/padded_noised_spliced_audio/")
                aa.write_audio_file(y_noise_pad,data_pad)









