# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 01:27:50 2019

@author: kbrya

https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
"""

#imports
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
df = pd.read_csv("iris.csv", header=None)
ds = df.values
X = ds[:,0:4].astype(float)
Y = ds[:,4]

#one hot encoding or creating dummy variables from a categorical variable
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

#creates a simple fully connected network
#Adam gradient descent optimization algorithm with a logarithmic loss function, “categorical_crossentropy”
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#create KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

#evaluate the neural network model on our training data
#shuffle before partitioning it 
kfold = KFold(n_splits=10, shuffle=True)

#k-fold cross validation, 10-fold cross-validation procedure
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#results are summarized as both the mean and standard deviation of the model accuracy on the dataset

"""
output

1)
Baseline: 96.67% (3.33%)

2)
Baseline: 98.00% (3.06%)

they take much longer than 10s on my laptop

"""