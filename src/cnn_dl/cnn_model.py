from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import librosa

import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DEBUG = True
EPOCHS = 15 

n_features = 20 # Number of features per time sample (row)
# n_label = -1 means the last column in the csv file
n_label = -1 # The column number that contains the labels
n_classes = 3 

word_size = 102 # 102 rows constitute one word
batch_size = 4096 #256  # 2^n words constitute the batch size
channels = 1 
# Flag to indicate whether to train a model or to load the weights from the saved models
load_data = 0
## Input Filenames
#train_filename = 'train_data_20features.csv'
train_filename = 'training_102x20.csv'
#test_filename = 'test_data_20features.csv'
test_filename = 'testing_102x20.csv'
## Saved Model Path and Filenames
model_filepath_loss = './models/best_model_loss_run11.h5'
model_filepath_accuracy = './models/best_model_loss_run11.h5'
model_filepath_saved = 'run11'

class params:
    sampling_rate = 22050 # Hz
    sample_duration = 2.35 # 1 word
    hop = 512 # window hop 
    time_frames = int(np.ceil((sample_duration * sampling_rate) / hop)) # 102
    batch_size = batch_size
    test_batch_size = 64
    n_mfcc = n_features
    channels = channels
    input_shape = (time_frames, n_mfcc, channels)
    classes_num = n_classes # Number of Classes
    # Training hyperparameters
    learning_rate = 1e-3
    epochs = EPOCHS
    

def get_model(params):
  input = tf.keras.Input(shape=params.input_shape)
  x = input
  x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  #x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 2), padding='SAME', activation='relu')(x)
  #x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 2), padding='SAME', activation='relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(x)

  x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  #x = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(x)
  #x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  #x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  #x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  #x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)
  #x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(x)

  x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
  #x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
  #x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
  #x =  tf.keras.layers.Reshape((x.shape[1], -1))(x)
  #x = tf.keras.layers.Flatten()(x)
  #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512, activation='tanh'))(x)
  #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512, activation='relu'))(x)
  #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='relu'))(x)
  #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='relu'))(x)
  x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(x)
  #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(x)
  x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(x)
  #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(x)
  x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(16, activation='relu')(x)

  x = tf.keras.layers.Dense(params.classes_num)(x)
  x = tf.keras.layers.Activation('softmax')(x)

  """
  x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(64, activation='tanh', dropout=0.1, recurrent_dropout=0.1, return_sequences=True),
            merge_mode='mul')(x)

  """
  model = tf.keras.Model(inputs=input, outputs=x)
  return model

# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def predict_accent(model, X_test, y_test):
    predictions = model.predict(X_test)
    return predictions

def train_model(X_train,y_train,X_validation,y_validation, batch_size=batch_size): 
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''
    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    # input image dimensions to feed into 2D ConvNet Input layer
    X_train = X_train.reshape(X_train.shape[0], rows, cols, channels )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols, channels)
    
    #############
    # Get Model #
    #############
    model = get_model(params)

    #############################
    # Compile Model and Summary #
    #############################
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    #accuracy = tf.keras.metrics.Accuracy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    #'sparse_categorical_crossentropy'
    model.compile(loss=loss, 
                    optimizer= optimizer,
                    metrics=['accuracy'])
    model.summary()
    #exit(0)
    
    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='accuracy', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    # Scheduler Callback
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_filepath_accuracy,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    best_loss_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_filepath_loss,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.001)
    if load_data == 0:
        # Fit model using ImageDataGenerator
        model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=int(len(X_train) // batch_size)
                        , epochs=EPOCHS,
                        callbacks=[tb, scheduler_callback, es, best_checkpoint_callback, best_loss_checkpoint_callback], validation_data=(X_validation,y_validation))
    else: 
        # Loads the weights
        model.load_weights(model_filepath_loss)

    return (model)

def save_model(model, model_filename):
    '''
    Save model to file
    :param model: Trained model to be saved
    :param model_filename: Filename
    :return: None
    '''
    #model.save('./models/{}.h5'.format(model_filename))  # creates a HDF5 file 'my_model.h5'
    model.save('./models/{}.h5'.format(model_filename))


if __name__ == '__main__':
    '''
    Console command example:
    python trainmodel.py bio_metadata.csv model50
    '''
    # Load Dataset CSV file
    if train_filename == 'dl_train.csv' or train_filename == 'dl_training.csv' or train_filename == 'training_102x20.csv':
        train = pd.read_csv(train_filename, header=None)
        test = pd.read_csv(test_filename, header=None)
    else: 
        train = pd.read_csv(train_filename)
        test = pd.read_csv(test_filename)
    # Locate MFCC features as our "X" inputs/attributes 
    # and labels "y" for the train and test datasets
    X_train = train.iloc[:,:n_features]
    y_train = train.iloc[:,n_label]
    X_test = test.iloc[:,:n_features]
    y_test = test.iloc[:,n_label]
    #X_train.head()
    #y_train.head()
    print(y_train.head().to_string())

    # Convert Panda Dataframes to Numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    if train_filename == 'train_data_20features.csv':
        # Map languages to integer categories
        y_train = np.where(y_train == 'Arabic', 0, y_train)
        y_train = np.where(y_train == 'Hindi', 1, y_train)
        y_train = np.where(y_train == 'Chinese', 2, y_train)
        print(y_train[0:5])
    
        y_test = np.where(y_test == 'Arabic', 0, y_test)
        y_test = np.where(y_test == 'Hindi', 1, y_test)
        y_test = np.where(y_test == 'Chinese', 2, y_test)
    elif train_filename == 'train_new_dataset.csv':
        # Map languages to integer categories
        y_train = np.where(y_train == 1, 0, y_train)
        y_train = np.where(y_train == 2, 1, y_train)
        y_train = np.where(y_train == 4, 2, y_train)
        print(y_train[0:5])
    
        y_test = np.where(y_test == 1, 0, y_test)
        y_test = np.where(y_test == 2, 1, y_test)
        y_test = np.where(y_test == 4, 2, y_test)
    elif train_filename == 'dl_train.csv' or train_filename == 'dl_training.csv' or train_filename == 'training_102x20.csv':
        # Map languages to integer categories
        y_train = np.where(y_train == 1, 0, y_train)
        y_train = np.where(y_train == 2, 1, y_train)
        y_train = np.where(y_train == 3, 2, y_train)
        print(y_train[0:5])
    
        y_test = np.where(y_test == 1, 0, y_test)
        y_test = np.where(y_test == 2, 1, y_test)
        y_test = np.where(y_test == 3, 2, y_test)

    # 177420 is the number of samples (sample being each time frame for each word for all audio file
    # Assuming 20 features
    # X_train is (177420, 20)
    # We want to reshape X to be 
    #(n_batches=n_rows/batch_size= 177420/64, batch_size=64, time_frames = 102, 20, 1) ## WRONG # Batches are handled by the dataframe generator
    #(n_rows/102=n_words, time_frames = 102, 20, 1) ## CORRECT
    ###################################################
    # Data Preprocessing: Reshaping and normalization #
    ###################################################
    n_rows = X_train.shape[0] # Number of samples (time frame for each word for all audio signals)
    n_samples = n_rows//word_size*word_size ## Number of samples divisible by the number of words (after discarding a few timeframes)
    X_train = X_train[:n_samples] # Discarding a number word time frames 
    n_words = int(n_samples/word_size) # Number of words in the dataset
    ##################
    # Data Reshaping #
    ##################
    X_train = X_train.reshape(n_words, word_size, n_features, 1) 
    #X_mean = np.mean(X_train, axis=(0,1)) # 20,1 
    #print(np.min(X_train), np.max(X_train)) # -12, 11 range
    
    # Test Data Reshaping
    n_rows_test = X_test.shape[0]
    n_samples_test = n_rows_test//word_size*word_size
    X_test = X_test[:n_samples_test]
    n_words_test = int(n_samples_test/word_size)
    X_test= X_test.reshape(n_words_test, word_size, n_features, 1)
    
    ######################
    # Data Normalization #
    ######################
    # TRAIN DATA NORMALIZATION

    # Get the mean of each MFCC column (mfcc0, mfcc1,..etc) in the original X_train (i.e. across axis 0 and axis 1 in the reshaped X_train)
    # Then add new axis so that the mean has the same shape as the reshaped X_train
    X_mean = np.mean(X_train, axis=(0,1))[np.newaxis, np.newaxis, :, :] # 1,1,20,1 
    # Similarly, get the standard deviation for each MFCC and reshape it
    X_std = np.std(X_train, axis=(0,1))[np.newaxis, np.newaxis, :, :] # 1,1,20,1 
    # Normalization
    #X_train = (X_train - X_mean)/X_std
    numerator = (X_train - X_mean)
    denominator = X_std
    X_train = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    print(np.min(X_train), np.max(X_train)) # -12, 11 range

    # The first derivative of the MFCC is mfcc_delta
    mfcc_delta  = librosa.feature.delta(np.squeeze(X_train,axis=-1), order = 1)[:,:,:,np.newaxis]
    # The second derivative of the MFCC is mfcc_delta2
    mfcc_delta2 = librosa.feature.delta(np.squeeze(mfcc_delta,axis=-1), order = 1)[:,:,:,np.newaxis]
    
    # TEST DATA NORMALIZATION
    X_mean_test = np.mean(X_test, axis=(0,1))[np.newaxis, np.newaxis, :, :] # 1,1,20,1 
    X_std_test = np.std(X_test, axis=(0,1))[np.newaxis, np.newaxis, :, :] # 1,1,20,1 
    X_test = (X_test - X_mean_test)/X_std_test
    mfcc_delta  = librosa.feature.delta(np.squeeze(X_test,axis=-1), order = 1)[:,:,:,np.newaxis]
    mfcc_delta2 = librosa.feature.delta(np.squeeze(mfcc_delta,axis=-1), order = 1)[:,:,:,np.newaxis]
    
    ##############################################
    # Concatenate the data for multiple channels #
    ##############################################
    if channels == 3: 
        #X_train = np.concatenate([X_train, mfcc_delta, mfcc_delta2], axis=-1)
        X_train = np.concatenate([X_train, X_train, X_train], axis=-1)
        #X_test = np.concatenate([X_test, mfcc_delta, mfcc_delta2], axis=-1)
        X_test = np.concatenate([X_test, X_test, X_test], axis=-1)

    ##################
    # Input Data Shuffling #
    ##################
    # Train Data Shuffling
    idx_list = np.arange(X_train.shape[0])
    seed = 0
    np.random.seed(seed)
    np.random.shuffle(idx_list)
    X_train = X_train[idx_list]

    #######################################
    # Label Data Rehshaping and Shuffling #
    #######################################
    y_train = y_train[:n_samples]
    y_train = y_train.reshape(n_words, word_size)
    y_train = y_train[:, 0]
    y_train = np.expand_dims(y_train, axis=1)
    y_train = y_train[idx_list]

    y_test = y_test[:n_samples_test]
    y_test = y_test.reshape(n_words_test, word_size)
    y_test = y_test[:, 0]
    y_test = np.expand_dims(y_test, axis=1)
    
    ###############
    ## CNN Model ##
    ###############

    # Train the model 
    model = train_model(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32))
    
    y_predicted = model.predict(X_test)
    y_pred = np.argmax(y_predicted, axis=1)[:, np.newaxis]
    
    # Save model
    save_model(model, model_filepath_saved)
    
    # Confusion Matrix, Accuracies and Classification Report
    def flatten(t):
        return [item for sublist in t for item in sublist]
    y_test = flatten(y_test)
    y_pred = flatten(y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
    print(" F1 score is %2.3f" % f1_score(y_test, y_pred,average='macro'))
    print(" Recall score  is %2.3f" % recall_score(y_test, y_pred,average='macro'))
    print(" Precision score  is %2.3f" % precision_score(y_test, y_pred,average='macro'))
    titles_options = [
    ("CNN", "true")
    ]
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize= 'true', 
        display_labels=['Arabic','Hindi','Chinese'], include_values=True, xticks_rotation='horizontal', values_format=None, 
        colorbar=True)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    plt.savefig('confusion_matrix_run11.png')

