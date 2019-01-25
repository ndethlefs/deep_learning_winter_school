import numpy as np
import sys
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.metrics import mean_squared_error
from keras import backend as K
import time
import keras
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from sklearn.model_selection import RandomizedSearchCV

x = []
y = []

num_classes = 4

for line in open('./wall_following_robot_sensors.txt', 'r'):
	l = line.split(',')
	inp = l[1:-1]
	outp = l[-1]	
	x.append([float(i) for i in inp])
	y.append(int(outp))
	

X_set = np.asarray(x)
Y_set = np.asarray(y)

print(X_set.shape)
print(Y_set.shape)

# this is a basic setup, depending on your model you may need to reshape...

start_time = time.time()

'''
Create x_train, x_test, y_train and y_test here

You will need these to train your model and then validate it on a separate dataset.

Best to shuffle all and then split e.g. 70/30, 80/20 or similar.

Once done, get rid of the below....

'''

x_train = X_set
x_test = X_set

y_train = Y_set
y_test = Y_set


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# the bit below defines the neural net... You can change that as much as needed. 

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(len(x[0]),)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adamax(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=5,
                    epochs=15,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




