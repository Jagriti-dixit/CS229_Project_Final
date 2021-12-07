## Overview ##

This is the code directory for our implementation of **two Classical ML models used as baseline techniques** for the Accent Recognition System.  

The techniques used for baselining are:  
**1. Gaussian Naive Bayes**  
**2. Logistic Regression**  

The code directory contains the following files and a directory:  
**1. accent_data.csv** - A metadata CSV file created to include the names of the speakers that we run the model on, and information related to the speakers and the audio files in the speaker directories.  
**2. accent_recognition.py** - This is the primary script for running the baseline models.   
**3. audio_preprocess_experiments.py** - This contains the code for padding the spliced audio words, adding noise to the spliced audio words and also padding the noise-added spliced words.  
**4. audio_preprocessing_helpers.py** - This contains the code for several helper methods used by accent_recognition.py and audio_preprocess_experiments.py. Helper methods include extracting MFCC features from an audio file, audio augmentation helper methods to pad, add noise to audio signals, read and write audio files, obtain the maximum duration from a directory containing audio files.  
**5. pca.py** - This code takes in an input csv and performs Principal Component Analysis for dimensionality reduction and produces an output csv for training and testing.
**6. getSamples.py** - Splices the sentences spoken by the speakers in the original dataset audio files to individual words.
**7. suitcase_suite** - The code directory for preprocessing and running the test on the suitcase_corpus subset of the dataset


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

Note: The dataset generation takes quite a while!
A link to the datasets we generated are:
1. Training - [train_new_dataset_final.csv](https://drive.google.com/file/d/1PJFAWSU6TPpueF0b5fMaZDI7V2i63Gvm/view?usp=sharing)
2. Testing - [test_new_dataset_final.csv](https://drive.google.com/file/d/1bn3gUej-k3OkpPl6ucs3aVzdEjiKmbuX/view?usp=sharing)
3. Suitcase Corpus Testing - [suitcase_test.csv](https://drive.google.com/file/d/1NXbeTS0j7c56cweYwuzXSLMiII-yNX1X/view?usp=sharing)


## Steps to Run the Model ##

1. There are several .wav files in each of the speaker directories. Each of these .wav audio files corresponds to a sentence spoken by the speaker. We must split these sentences into individual words. Run the below command to populate spliced word audio files within the speaker directories.

```
python3 getSamples.py
```

2. These spliced wav files representing a word are padded to ensure that the dimensions of the obtained MFCC representations across all words are uniform. Further, to augment the data we add noise to the spliced wav files and padd these files. To pad and add noise, the following python code is run:  

```
python3 audio_preprocess_experiments.py
```

3. After the data preprocessing, we generate the design matrix for the model training and testing and run the models on the CSV files obtained. (The very first run should always use **--create** to generate the dataset CSVs (for testing and training).

```
python3 accent_recognition.py --create <training file> <testing file>
```
  
For the subsequent runs (or when the test and train CSVs are readily available), we can choose to just load the CSVs and run the model on them:

```
python3 accent_recognition.py --load <training file> <testing file>
```

4. Dimensionality Reduction using PCA - In case we wish to perform PCA reduction on the training and testing CSV file we can use the **pca.py** script:
```
python3 pca.py <input_train_file> <input_test_file>
```

## Suitcase Corpus ##
This portion of the L2-ARCTIC corpus involves spontaneous speech. We use this an additional testing dataset for our model. Please refer to the README.md inside the suitcase_suite directory for additional details.
