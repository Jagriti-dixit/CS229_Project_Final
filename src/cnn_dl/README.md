## Overview ##

The CNN implementation is implemented in cnn_model.py. The CNN architecture is described below:  
```

Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 102, 20, 1)]      0         
_______________________
conv2d (Conv2D)              (None, 102, 20, 32)       320       
_______________________
conv2d_1 (Conv2D)            (None, 102, 20, 64)       18496     
_______________________
conv2d_2 (Conv2D)            (None, 102, 10, 64)       36928     
_______________________
average_pooling2d (AveragePo (None, 102, 5, 64)        0         
_______________________
conv2d_3 (Conv2D)            (None, 102, 5, 128)       73856     
_______________________
conv2d_4 (Conv2D)            (None, 102, 5, 128)       147584    
_______________________
conv2d_5 (Conv2D)            (None, 102, 5, 256)       295168    
_______________________
max_pooling2d (MaxPooling2D) (None, 51, 2, 256)        0         
_______________________
time_distributed (TimeDistri (None, 51, 2, 128)        32896     
_______________________
time_distributed_1 (TimeDist (None, 51, 2, 128)        16512     
_______________________
time_distributed_2 (TimeDist (None, 51, 2, 64)         8256      
_______________________
time_distributed_3 (TimeDist (None, 51, 2, 32)         2080      
_______________________
flatten (Flatten)            (None, 3264)              0         
_______________________
dense_4 (Dense)              (None, 16)                52240     
_______________________
dense_5 (Dense)              (None, 3)                 51        
_______________________
activation (Activation)      (None, 3)                 0         
=================================================================
Total params: 684,387
Trainable params: 684,387
Non-trainable params: 0
```

The code directory includes the following directories and files
    1. ./src/cnn_model.py – This is the primary script for training or loading the CNN model and for inference.
    2. ./models/ - This directory includes saved models from the best results callbacks (including best accuracy and least loss).
    3. ./logs/ – This directory is the output of the tensorboard loggings for a visual graphical interpretation of the run.
    
    
## Environment Setup ##
Installation of tensorflow and tensorboard in addition to classical ML libraries such as Librosa and Scikit-Learn.

## Datasets ##
The datasets were obtained by running the following helper python scripts:

```
python3 audio_preprocessing_dl.py
```

For the suitcase_corpus test dataset generation:
```
python3 audio_preprocessing_dl_suitcase.py
```


Note: The dataset generation takes quite a while!
A link to the datasets we generated are:
1. Training - [training_102x20.csv](https://drive.google.com/file/d/1QYS2nsAHJcPcTEhIpo_-JR6irgGj8YcQ/view?usp=sharing)
2. Testing - [testing_102x20.csv](https://drive.google.com/file/d/1fKLmSCZFzftkexYhZL6sQduAIS6HkGDY/view?usp=sharing)
3. Suitcase Corpus Testing - [suitcase_corpus_test_102x20.csv](https://drive.google.com/file/d/10eWpdF2MeupwbSnv9nvn-z5cmcfacUC3/view?usp=sharing)


## Steps to Run the Model ##
**1. Adjust the input parameters in cnn_model.py:**   
a. EPOCHS is the maximum number of epochs to be run.  
b. n_features is the number of MFCC features.  
c. n_label is the column number in the input CSV files which includes the labels.  
d. word_size is the number of time frames of each word.  
e. batch_size is a hyperparameter that describes the number of words to be processed during each iteration within each epoch. The number of iterations in each epoch is the number of words divided by the batch_size.  
f. n_classes is the number of classes. Our run includes the Arabic, Hindi and Chinese accents in the dataset. Hence, the default n_classes is set to three.  
g. channels is the number of channels to defined for the CNN model.  
h. Input filenames: trian_filename and test_filename for the training and test CSV files resulted from the feature extraction part of the code.   
i. Model filenames: model_filepath_loss and model_filepath_accuracy are the respective filenames for the saved models for least loss and best accuracy during the run.  
j. load_data is a flag to indicate whether to load the weights from the latest saved model (if set to 1) or to train a new model (if set to zero).  

Note: The params class also includes sampling_rate, sample_duration, window hop (hop) and initial learning rate which may be adjusted depending on the audio signal and desired learning rate.  

**2. Run the model:** 
```
python3 cnn_model.py
```

**3. Viewing the Output:**
The output log text on the terminal will include the model summary, and the ETA of each epoch, the training and validation accuracy of each epoch, and the early stopping condition when met. When the run is complete, test accuracies of the model will be shown based on the F1-score, precision and recall scores. In addition, a confusion matrix will appear with the validation/test results for each accent class.  

**4. Viewing the Logs:**
To view the tensorboard log files, go to the directory containing the “logs” directory and type on the command line: 
```
tensorboard –logdir logs  
```  
   
**5. Viewing the Graphs:**   
Now, you can view the loss-vs-epochs and accuracy graphs on the indicated localhost ip via any browser (for example: http://localhost:6006/). To exit, close the browser window and click CTRL+C in the terminal to exit tensorboard.
