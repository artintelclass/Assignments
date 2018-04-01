# Heart Diseae Data Training Exercise

## Dividing and cleaning the data
In the original file, there are 303 valid datasets, and each of them includes 14 numbers, showing 13 different variables (inputs) and 1 result (ouput). 
The datasets were divided into two groups by separating them in two csv file, where they have 250 and 53 data, individually. The first group of data is used to trian the model, and the second one is used to evaluate it.
The original datasets include 6 empty valued denoted by "?", and therefore were replaced by the rounded average value of corresponding column.

## Reading and setting the data
After loading the data with pandas, the last column was separated as the label of results, and the rest columns were applied as numpy arrays. In original files, the last column's value dataset are denoted by 0 to 4, while the model only need 2 classes, here we used "clip()" function to change the data value to be 0 or 1.

## Training data
Based on certain training and valuating samples, the variable parameters we can modify are list as below:
1. batch size, epochs.
2. activation functions (such as relu, tanh, sigmoid, etc.)
3. dropout rate.
3. optimizers (such as SGD, Adam, RMSprop, etc.)

Trying different matches, finnaly I made a 81.13% accuracy with batch_size=10, epochs=400, 2 layers of relu activation fuction followed by 0.25 dropout rate, and 1 layer of softmax activation functio for output.
(result.png)
However, there is still some turbulation on results, and usually the accuracy cannot reach so high but around 75% accuracy.

## [Code](yuxin_heartdisease_exercise.ipynb)

```python
from __future__ import print_function
import tensorflow.contrib.keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.optimizers import RMSprop,SGD,Adam
import pandas as pd
import numpy as np


#overal, based on the same algorithm, what influence the learning is the bactch, epoches and hiddenlayers and nodes.
batch_size = 10#updating times= sample/batch*epochs
num_classes = 2#the number of classification
epochs = 400
# use pandas to read files
# we don't have a header, and we don't want the first column to be index values
training = pd.read_csv("train.csv", header=None, index_col=False)
# take the last column and store as labels as numpy array
y_train= np.asfarray(training.iloc[:,13])
# delete the last column and use the rest as data as numpy array
x_train= np.asfarray(training.drop(columns=[13]))
y_train = np.clip(y_train,0,1)
# use pandas to read file we don't have a header, and we don't want the first column to be index values
test = pd.read_csv("test.csv", header=None, index_col=False)
# take the first column and store as labels as numpy array
y_test= np.asfarray(test.iloc[:,13])
# delete the first column and use the rest as data as numpy array
x_test= np.asfarray(test.drop(columns=[13]))
y_test = np.clip(y_test,0,1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(13,)))#the number of features
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.summary()#print out function
# For a binary classification problem
model.compile(loss='categorical_crossentropy', optimizer=RMSprop() ,metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test,verbose=0)#should we fill the batch size here?
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

