import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

metadata_file = './accent_data.csv'

#Returns a MFCC array for the audio file containing a word utterance that are padded to equal lengths
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
 


##Audio augmentation- Adding noise,shifting, padding, stretching the audio signals. In interest of time for training the model have used only noise and padding for milestone submission

class AudioAugmentation:
    def read_audio_file(self, file_path):
        data = librosa.core.load(file_path)[0]
        return data

    def write_audio_file(self, file, data, sample_rate=22050):
        sf.write(file, data, sample_rate)

    def plot_time_series(self, data):
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()

    def add_noise(self, data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005* noise
        return data_noise

    def shift(self, data):
        return np.roll(data, 22050)

    def stretch(self, data, rate):
        data = librosa.effects.time_stretch(data, rate)
        return data

    
    def pad(self,data,max_value):
        pad_value = int(max_value*22050)
        if len(data) < pad_value:
            data_pad = np.pad(data, (0, max(0, pad_value - len(data))), "constant")
        return data_pad
    
    def getMaxLengthAudio(self, audio_dir):
        duration = []
        wav_name = []
        speaker = []
        df = pd.read_csv(metadata_file)
        for i in range(df.shape[0]):
            pathAudio = './' + df.loc[i].at['Speaker'] + '/spliced_audio/'
            audio_files = librosa.util.find_files(pathAudio, ext=['wav']) 
            files = np.asarray(audio_files)
            for y in files:
                audio,sample_rate = librosa.load (y,res_type = 'kaiser_fast')
                t = librosa.get_duration(y=audio,sr=sample_rate)
                duration.append(t)
                wav_name.append(y)
        
        t_max = max(duration)
        idx = duration.index(t_max)
        audio_name_max = wav_name[idx]

        print("\nThe maximum length is ",len(data))
        print("\nThe audio file with maximum length is ", audio_name_max)
