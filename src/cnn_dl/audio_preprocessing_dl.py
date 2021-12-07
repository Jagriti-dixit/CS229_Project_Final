import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys
import random

#actual dataset reading
metadata_file = './accent_data.csv'
dirPath = "/home/paavni/processed_data/"
#Check if the file exists that can be loaded to the dataFrame 
df = pd.read_csv(metadata_file)
df['relative_path'] =  df['Speaker'].astype(str) 
df = df[['relative_path','Speaker','Gender','Native_Language','Data_Type']]

#Returns a MFCC array for the audio file containing a word utterance that are padded to equal lengths
def extract_features(file_name, speaker, gender, language):
    #Load the audio file
    audio,sample_rate = librosa.load(file_name,res_type = 'kaiser_fast')
    #Obtain the MFCC array for each such word utterance
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate)
    #Normalize the MFCC array by rows
    mfcc_normalized = mfcc/np.linalg.norm(mfcc, ord=2, axis=1, keepdims=True)
    mfcc_normalized_t = np.transpose(mfcc_normalized)
    arr = np.full((mfcc_normalized_t.shape[0], 1), speaker)
    arr = np.append(arr, np.full((mfcc_normalized_t.shape[0], 1), gender), axis=1)
    arr = np.append(arr, np.full((mfcc_normalized_t.shape[0], 1), language), axis=1)
    features = np.append(mfcc_normalized_t, arr, axis=1)
    #print("Shape of the matrix: ", features.shape, "\n")
    return features   

if __name__ == "__main__":
    if(len(sys.argv)!=3):
        raise Exception("Incorrect number of arguments! Usage: audio_preprocessing_dl.py <training file> <testing file>")
    else:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        files_all = []

        #Create a training CSV file of the entire dataset with all the features
        print("===============================Extracting features===========================\n")
        #Assigning values for classification to gender and language
        #Populate the list of files for each of the speakers
        for i in range(df.shape[0]):
            audio_files = []
            audio_noise_files = []
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

            audioPath = dirPath + speaker_name + '/padded_spliced_audio/'
            audioPath_noise = dirPath + speaker_name + '/padded_noised_spliced_audio/'
            audio_files.extend(librosa.util.find_files(audioPath, ext=['wav']))
            audio_noise_files.extend(librosa.util.find_files(audioPath_noise, ext=['wav']))
            audio_files.extend(audio_noise_files)
            print(len(audio_files))
            print(audio_files[0])
            #Iterate through the files and construct the files_all list
            for i in range(len(audio_files)):
                files_all.append([audio_files[i], speaker_name, a, b, data_type])
            
        #Shuffle the write-outs of the words in the training files for the 3 speakers
        random.shuffle(files_all)
        #print(len(files_all))
            
        for file_n in files_all: 
            # print("file is", file)
            features = extract_features(file_n[0], file_n[1], file_n[2], file_n[3])
            #print("\nFeature Matrix Shape for ", speaker_name, " is: ", features.shape, "\n")
            # Write this data to the CSV file (either test or train csv file according to the speaker's data_type in metadata file)
            df_csv = pd.DataFrame(data=features)

            if(file_n[4] == "Train"):
                df_csv.to_csv(train_file, index=False, header=False, mode = 'a')
                print("\nSuccessfully Added Information for: ", file_n[1], " file: ", file_n[0], " to the Training File\n")
            else:
                df_csv.to_csv(test_file, index=False, header=False, mode = 'a')
                print("\nSuccessfully Added Information for: ", file_n[1], " file: ", file_n[0], " to the Testing File\n")         





               

           
            
            
