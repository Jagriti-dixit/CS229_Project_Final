import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys
import random

metadata_file = './accent_data.csv'
#Check if the file exists that can be loaded to the dataFrame 
df = pd.read_csv(metadata_file)
df['relative_path'] =  df['Speaker'].astype(str) 
df = df[['relative_path','Speaker','Gender','Native_Language']]

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
    if(len(sys.argv)!=2):
        raise Exception("Incorrect number of arguments! Usage: audio_preprocess_dl.py <testing file>")
    else:
        test_file = sys.argv[1]
        files_all = []

        #Create a CSV file used for testing of the entire dataset with all the features
        print("===============================Extracting features===========================\n")
        #Assigning values for classification to gender and language
        #Populate the list of files for each of the speakers
        for i in range(df.shape[0]):
            matches = []
            matches_noise = []
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
            matches = [os.path.abspath(pathOutput + '/' + fname) for fname in os.listdir(pathOutput) if fname.startswith(df.loc[i].at['Speaker'])]
            matches_noise = [os.path.abspath(pathOutput_noised + '/' + fname) for fname in os.listdir(pathOutput) if fname.startswith(df.loc[i].at['Speaker'])]
            matches.extend(matches_noise)
            #print(len(matches))
            #print(matches[0])
            #Iterate through the files and construct the files_all list
            for i in range(len(matches)):
                files_all.append([matches[i], speaker_name, a, b])
            
        #Shuffle the write-outs of the words in the training files for the 3 speakers
        random.shuffle(files_all)
        #print(len(files_all))
            
        for file_n in files_all: 
            # print("file is", file)
            features = extract_features(file_n[0], file_n[1], file_n[2], file_n[3])
            #print("\nFeature Matrix Shape for ", speaker_name, " is: ", features.shape, "\n")
            # Write this data to the test CSV file 
            df_csv = pd.DataFrame(data=features)        
            df_csv.to_csv(test_file, index=False, header=False, mode = 'a')
            print("\nSuccessfully Added Information for: ", file_n[1], " file: ", file_n[0], " to the Testing File\n")         