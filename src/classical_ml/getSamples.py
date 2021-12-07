################################This file is for splicing words from sentences###########################
import os
from os.path import join
import pandas as pd
import librosa

from praatio import textgrid
from praatio import audio
from praatio import praatio_scripts
#from playsound import playsound

#For testing the script
# audioFile = "lxc_arctic_a0018.wav"
# tgFile = r"lxc_arctic_a0018_textgrid.Textgrid"

pwdPath = os.getcwd()

def getWords(inDirPath, audioFile, outDir):
    audioPath = os.path.join(inDirPath, "wav")
    tgPath = os.path.join(inDirPath, "textgrid")
    #print(tgPath)
    if(os.path.isdir(audioPath)==False):
        raise Exception("No such audio file directory: ", audioPath)
    if(os.path.isdir(tgPath)==False):
        raise Exception("No such audio file directory: ", tgPath)
    
    tierName = "words"
    sourceAudioFileName = os.path.relpath(audioFile, audioPath)
    tgFileName = sourceAudioFileName[:-4]+".TextGrid"
    tgFile = os.path.join(tgPath, tgFileName)
    #print(tgFile)
    if((os.path.isfile(tgFile))==False):
        raise Exception("No such TextGrid File: ", tgFileName)
    sourceAudioObj = audio.openAudioFile(audioFile)
    tg = textgrid.openTextgrid(tgFile, False)
    tier = tg.tierDict[tierName]
    for i in range(0, len(tier.entryList)):
        #print(tier.entryList[i])
        x_min = tier.entryList[i][0]
        x_max = tier.entryList[i][1]
        #print(x_min, x_max, audioFile, tgFile)

        splicedAudio = sourceAudioObj.getSubsegment(x_min, x_max)
        splicedFileName = sourceAudioFileName[:-4]+"_"+str(i)+".wav"
        #print(splicedFileName)
        outputFile = os.path.join(outDir, splicedFileName)
        #print(outputFile)
        splicedAudio.save(outputFile)
        # playsound(outputFile)


def performAudioSplicing():
    metadata_file = os.path.join(pwdPath, "accent_data.csv")

    #Check if the file exists that can be loaded to the dataFrame 

    df = pd.read_csv(metadata_file)
    speakers = df['Speaker'].tolist()
    #print(speakers)

    # Check if that directory is present
    for speaker in speakers:
        if(os.path.isdir(speaker)):
            #Set the wav directory and texgrid directory paths
            dirPath = os.path.join(pwdPath, speaker)
            wavPath = os.path.join(pwdPath, speaker, "wav")
            #Set the output path for all the spliced files and create the directory
            outPath = os.path.join(pwdPath, speaker, "spliced_audio")
            if(os.path.isdir(outPath)==False):
                os.mkdir(outPath)
                #print(outPath)
            
            #Perform splicing of all the audio files for the particular speaker
            audio_files = librosa.util.find_files(wavPath, ext=['wav']) 
            print(audio_files[0])
            print(speaker, len(audio_files))

            #Parse through all the wav files and textgrid files and populated the "spliced_audio" directory
            for file_a in audio_files:
                getWords(dirPath, file_a, outPath)
            
            print("File processing complete for: ", speaker, outPath)
    

#For testing the script
if __name__ == "__main__":
    performAudioSplicing()
