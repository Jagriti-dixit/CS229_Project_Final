import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

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
            pad_data = np.pad(data, (0, max(0, pad_value - len(data))), "constant")
        return pad_data
