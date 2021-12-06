This is the code directory for our implementation of **two Classical ML models used as baseline techniques** for the Accent Recognition System.

The techniques used for baselining are:
**1. Gaussian Naive Bayes**
**2. Logistic Regression**


The code directory contains the following files:
**1. accent_recognition.py** - This is the primary script for running the baseline models 
**2. audio_preprocess_experiments.py** - This contains the code for padding the spliced audio words, adding noise to the spliced audio words and also padding the noise-added spliced words
**3. audio_preprocessing_helpers.py** - This contains the code for several helper methods used by accent_recognition.py and audio_preprocess_experiments.py. Helper methods include
                                    extracting MFCC features from an audio file, audio augmentation helper methods to pad, add noise to audio signals, read and write audio files,
                                    obtain the maximum duration from a directory containing audio files.
**4. getSamples.py** - Splices the sentences spoken by the speakers in the original dataset audio files to individual word utterances



                                 
