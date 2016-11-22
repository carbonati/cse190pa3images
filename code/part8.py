import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,Cropping2D ,ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.datasets import cifar100
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import Tkinter as tk

weights_path = '8a_cifar100.h5'

#img_width, img_height = 32, 32

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
(X10_train, y10_train), (X10_test, y10_test) = cifar10.load_data()

X_train = X_train.astype('float')
X_test = X_test.astype('float')
X10_train = X10_train.astype('float')
X10_test = X10_test.astype('float')




# one-hot encoding for 100 classes
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y10_train = np_utils.to_categorical(y10_train)
y10_test  = np_utils.to_categorical(y10_test)

nb_classes = len(y_test[1])

model = Sequential()

#model.load_weights('8a_cifar100.h5', by_name=True)

model.add(Convolution2D(32, 3, 3, input_shape=(32,32,3),  activation='relu', border_mode = 'same'))
model.add(Cropping2D(cropping=((2,2),(2,2))))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.5))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1028, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(1028, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.6))
model.add(Dense(nb_classes, activation='softmax'))

print(model.summary())

#model.load_weights('8a_cifar100.h5', by_name=True)
#top_model.load_weights('hahadense.h5')

#model.add(top_model)
'''
#for layer in model.layers[:15]:
#	layer.trainable=False
# model.load_weights('haha.h5')
'''
print "number of layers l o l: ", len(model.layers)


epochs = 25
l_rate = 0.01
decay = l_rate/epochs
sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',
		optimizer=sgd,
		metrics=['accuracy'])

#hist = model.fit(X10_train, y10_train, batch_size = 64, nb_epoch=5, validation_data=(X10_test, y10_test))

#scores = model.evaluate(X10_test, y10_test, verbose=0)
hist = model.fit(X_train, y_train, batch_size=64, nb_epoch=5, validation_data=(X_test, y_test))
scores = model.evaluate(X_test, y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1] * 100)


model.save_weights('cifar100_5epochs.h5')

#model.save_weights('hahadense.h5')



print "Done with cifar100 weight training, now let's fine tune to fit cifar10 model"

cf10 = Sequential()
cf10.add(Convolution2D(32, 3,3, input_shape=(32, 32, 3), activation='relu', border_mode='same'))
cf10.add(Cropping2D(cropping=((2,2),(2,2))))
cf10.add(MaxPooling2D(pool_size=(2,2)))
cf10.add(Convolution2D(128,3,3, activation='relu', border_mode='same'))
cf10.add(MaxPooling2D(pool_size=(2,2)))
cf10.add(Convolution2D(256, 3,3, activation='relu', border_mode='same'))
cf10.add(Dropout(0.5))
cf10.add(Convolution2D(256, 3,3, activation='relu', border_mode='same'))
cf10.add(MaxPooling2D(pool_size=(2,2)))
cf10.add(Flatten())
cf10.add(Dense(1028, activation='relu', W_constraint=maxnorm(3)))
cf10.add(Dropout(0.5))
cf10.add(Dense(1028, activation='relu', W_constraint=maxnorm(3)))
cf10.add(Dropout(0.6))
cf10.add(Dense(10, activation='softmax'))

for i in range(len(cf10.layers)-6):
	cf10.layers[i].set_weights(model.layers[i].get_weights())



print (cf10.summary())

cf10.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = cf10.fit(X10_train, y10_train, batch_size=64, nb_epoch=5, validation_data=(X10_test, y10_test))
scorescf10 = model.evaluate(X10_test, y10_test, verbose=0)
print 'accuracy: %.2f%%' % (scorescf10[1] * 100)

cf10.save_weights('cf10_5epochs.h5')

# plots of accuracy and loss
f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(hist.history['loss'])
ax1.plot(hist.history['val_loss'])
ax1.set_title('Fine Tuned Loss Rate cifar10')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Training interations')
ax1.legend(['Training', 'Testing'], loc='upper left')
plt.show()
f1.savefig('Part8_Loss_rate1.png')

ax2 = f2.add_subplot(111)
ax2.plot(hist.history['val_acc'])
ax2.set_title('Fine Tuned Accuracy Rate cifar10')
ax2.set_ylabel('Accuracy %')
ax2.set_xlabel('Training iterations')
ax2.legend(['Testing'], loc='upper left')
plt.show()
f2.savefig('Part8_Accuracy_plot1.png')













