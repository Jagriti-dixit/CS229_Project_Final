This is the code directory for preprocessing and testing the two Classical ML models used as baseline techniques on the suitcase_corpus subset of the L2 arctic dataset.

The code directory contains the following files:
**1. accent_recognition_suitcase.py** - This is the primary script for running the baseline models.
**2. audio_preprocess_experiments.py** - This contains the code for padding the spliced audio words, adding noise to the spliced audio words and also padding the noise-added spliced words.
**3. audio_preprocessing_helpers.py** - This contains the code for several helper methods used by accent_recognition.py and audio_preprocess_experiments.py. Helper methods include extracting MFCC features from an audio file, audio augmentation helper methods to pad, add noise to audio signals, read and write audio files, obtain the maximum duration from a directory containing audio files.
**4. getSamples.py** - Splices the sentences spoken by the speakers in the original dataset audio files to individual words. 

## Dataset Details ##
Extract the **suitcase_corpus** directory present in the L2 arctic dataset to our **suitcase_suite** directory.

## Steps to Run the Model ##

1. There is a **SINGLE** .wav file corresponding to each speaker in the **suitcase_corpus** directory. Each of these .wav audio files corresponds to a sentence spoken by the speaker. We must split these sentences into individual words. Run the below command to populate spliced word audio files within the **suitcase_corpus** directory.

```
python3 getSamples.py
```

2. These spliced wav files representing a word are padded to ensure that the dimensions of the obtained MFCC representations across all words are uniform. Further, to augment the data we add noise to the spliced wav files and padd these files. To pad and add noise, the following python code is run:  

```
python3 audio_preprocess_experiments.py
```

3. We convert the spliced audio files to corresponding MFCCs and create a test CSV file. (The very first run should always use **--create** to generate the **test dataset CSV**). We **use the training CSV file obtained from the larger subset of speaker directories** (when we run the model in classical_ml directory)

```
python3 accent_recognition_suitcase.py --create <training file> <testing file>
```
  
For the subsequent runs (or when the test CSV is readily available), we can choose to just load the CSVs and run the model on them:

```
python3 accent_recognition.py --load <training file> <testing file>
```
