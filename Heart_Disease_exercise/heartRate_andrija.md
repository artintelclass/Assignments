# Simple NN

## Cleaning the data

The dataset had some entries with '?' which denoted data that was missing in the original entries. In order to be able to use the data, I had to remove this. I replaced all instances of '?' with the average data value for that particualr feature. I did this in Excel, by calculating and average and replacing. 

## Reading / interpreting the data

This involved a few steps. 
1. Insantiating the .csv data as a pandas object
2. Splitting the entire data set into 2 separate training and test sets
3. Within each of these sets, splitting between the features [0:12] and the label [13].
4. Converting these to numpy arrays as this is what the ML implementation works with
5. The labels range from (0-5), 0 being no heart disease, and 1-5 being various degrees of risk. As I was only concerned with their being none or any risk at all, I clipped the label values to 1, so that anything larger than 1 would become 1. 

## Adjusting the Learning model to improve the accuracy

Without making any changes, I had an accuracy of 0.23, or 23%.

Reading through the [Keras documentation](https://keras.io) was very helpful for figuring out which arguments to change and how.

Some of the changes that I made inlcuded:
1. Altering the batch size, and num_epochs.
2. Trying different activation functions (softplus, softsign, sigmoid, hard_sigmoid)
3. Changing the optimizer in model.complile() (Adadelta etc.)
4. Changing the optimizers learning rate 
5. Trying different loss functions

Ultimately I settled with the parameters in the code below. Although not consistently, running this training and testing gave up to 93% accuracy. 

## My code

You can grab the code from [here](andrija_Keras_fast.ipynb).

```python
'''Trains a simple deep NN on Cleveland heart disease dataset.
Gets to 93% test accuracy after 6 epochs
'''
import pandas as pd
import numpy as np

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential

from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.optimizers import Adadelta


batch_size = 110
# num_classes set to 2, in order to distinguish between 0 and everything else (>= 0)
num_classes = 2
epochs = 6

# read the data.csv file into a pandas object, telling it that there is no header, and no index column.
cardio = pd.read_csv('cleveland.csv', header=None, index_col=False)

# the size of an entry, from the .shape() method from pandas
# [0,1] returns the length, height of the array
size = cardio.shape[0]
# determine how to split the data between a training set and a test set
train_size = 290
test_size = 13

# train_data = cardio.iloc(0:290,0:14)
training_labels =np.asfarray(cardio.iloc[:train_size,13])
training_features = np.asfarray(cardio.iloc[:train_size,0:13])

test_labels = np.asfarray(cardio.iloc[train_size:size,13])
test_features = np.asfarray(cardio.iloc[train_size:size,0:13])

x_train = training_features
# for the labels, clip between 0 and 1, since we're not concerned with 1-5, only 0 and >0.
y_train = np.clip(training_labels,0,1)
x_test = test_features
y_test = np.clip(test_labels,0,1)


"""# these lines are not necessary, since we aren't using the mnist data set
# and since we do not need to reshape the arrays, or scale the values between 0. and 1. 

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
"""

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(13,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# model.add(Dense(num_classes, activation='relu'))



model.summary()

model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
              optimizer=Adadelta(lr=0.05),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
