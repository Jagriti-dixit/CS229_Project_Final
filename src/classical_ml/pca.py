import sys
import time
from comet_ml import Experiment
import pydub
import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import math,random
import zipfile as zf
import soundfile as sf
import pandas as pd
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import json
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import getSamples as gs
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
train_file = sys.argv[1]
test_file = sys.argv[2]

print("Reading train and test dataset")
#train = pd.read_csv('train_data_noise_pad.csv')
train = pd.read_csv(train_file)
print("read train data")
#test = pd.read_csv('test_data_noise_pad.csv')
test = pd.read_csv(test_file)
print("read test data")
print("Read two big files ")
X_train = train.iloc[:,:2040]
y_train = train.iloc[:,2041]
X_test = test.iloc[:,:2040]
y_test = test.iloc[:,2041]
# X_train = train.iloc[:,:20]
# y_train = train.iloc[:,21]
# X_test = test.iloc[:,:20]
# y_test = test.iloc[:,21]
X_train = StandardScaler(with_mean=True).fit_transform(X_train)
X_test = StandardScaler(with_mean=True).fit_transform(X_test)
print("Mean of train data is ",np.mean(X_train),"Std deviation is",np.std(X_train))
pca = PCA(n_components = 'mle')
pca = PCA().fit(X_train)
print('Explained variation per principal component:{}'.format((pca.explained_variance_ratio_)))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('Cumulative explained variance')
plt.savefig("cumulative_variance_plot.png")

time_start = time.time()
print("we want to see the accumulated variance of 700 features ")
pca = PCA(n_components = 700)
pca_result = pca.fit_transform(X_train)
pca_test = pca.transform(X_test)
X_train_pca = pca_result
X_test_pca = pca_test

out_train = "train_pca.csv"
pca_train = pd.DataFrame(data=X_train_pca)
pca_train['language'] = y_train
out_file_train = open(out_train,'wb')
pca_train.to_csv(out_file_train,index=False)
out_file_train.close()

out_test = "test_pca.csv"
pca_test = pd.DataFrame(data=X_test_pca)
pca_test['language'] = y_test
out_file_test = open(out_test,'wb')
pca_test.to_csv(out_file_test,index=False)
out_file_test.close()


time_start = time.time()


print("shapes are",X_train_pca.shape,y_train.shape)

print("X_train shape is ",X_train_pca.shape,"X_test shape is",X_test_pca.shape)
print("Total variation in these 1000 features is",np.sum(pca.explained_variance_ratio_))
print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
print("Now lets plot PCA for 2D visualisation")

##Taking only some of the total dataset randomly for plotting

np.random.seed(42)
rndperm = np.random.permutation(train.shape[0])

#2D plot(Having two components)
plt.figure(figsize=(16,10))

pca = PCA(n_components = 2)
pca_result = pca.fit_transform(X_train)
train['pca_one'] = pca_result[:,0]
train['pca_two'] = pca_result[:,1]
sns.scatterplot(
    x="pca_one", y="pca_two",
    hue="2041",
    palette=sns.color_palette("hls", 3),
    data=train.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
plt.savefig("PCA_2d.png")

###PCA with 3 components
pca = PCA(n_components = 3)
pca_result = pca.fit_transform(X_train)
train['pca_one'] = pca_result[:,0]
train['pca_two'] = pca_result[:,1]
train['pca_three'] = pca_result[:,2]
print("Its processing 3d plot")



#3D plot(Having 3 components)
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=train.loc[rndperm,:]["pca_one"], 
    ys=train.loc[rndperm,:]["pca_two"], 
    zs=train.loc[rndperm,:]["pca_three"], 
    c=train.loc[rndperm,:]["2041"], 
    cmap='tab10'
)
ax.set_xlabel('pca_one')
ax.set_ylabel('pca_two')
ax.set_zlabel('pca_three')
plt.savefig("PCA_3d.png")
