## Overview ##

This is the code directory for our implementation of **two Classical ML models used as baseline techniques** for the Accent Recognition System.  

The techniques used for baselining are:  
**1. Gaussian Naive Bayes**  
**2. Logistic Regression**  

The code directory contains the following files:  
**1. accent_recognition.py** - This is the primary script for running the baseline models.   
**2. audio_preprocess_experiments.py** - This contains the code for padding the spliced audio words, adding noise to the spliced audio words and also padding the noise-added spliced words.  
**3. audio_preprocessing_helpers.py** - This contains the code for several helper methods used by accent_recognition.py and audio_preprocess_experiments.py. Helper methods include extracting MFCC features from an audio file, audio augmentation helper methods to pad, add noise to audio signals, read and write audio files, obtain the maximum duration from a directory containing audio files.  
**4. getSamples.py** - Splices the sentences spoken by the speakers in the original dataset audio files to individual words.


## Environment Setup ##

An installation of Python3 is required for the project. 

Run the commands below to install the necessary Python3 libraries:

```
pip3 install numpy pandas librosa praatio soundfile matplotlib comet_ml pydub scikit-learn pathlib
```

## Dataset Details ##
For this project, the [L2-ARCTIC corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/) was used.This is a non-native (or L2) English speech corpus that is
intended for research in voice conversion, accent conversion, and mispronunciation detection. In total, the corpus contains 26,867 utterances from 24 non-native speakers with a balanced gender for six different languages: Arabic, Mandarin, Hindi,Korean, Spanish and Vietnamese.

**We make use of a smaller subset of this dataset that corresponds to the Arabic, Hindi and Chinese language for training and testing our accent recognition system**

Note: The dataset has to be obtained by a request on the link provided. 

**We unzip and place the directories correspoding to the speakers (such as "ABA", "BWC") within the classical_ml directory.**  


## Steps to Run the Model ##

1. There are several .wav files in each of the speaker directories. Each of these .wav audio files corresponds to a sentence spoken by the speaker. We must split these sentences into individual words. Run the below command to populate spliced word audio files within the speaker directories.

```
python3 getSamples.py
```
