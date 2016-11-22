# THIS IS FOR QUESTION 6b

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import Tkinter as tk


(X_train, y_train), (X_test, y_test) = cifar100.load_data()

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
"""
# takes the first 100 indexes of each class
train_indexes = [np.where(y_train==i)[0][:300] for i in range(100)]
indexes = []
for i in range(100):
    for j in range(300):
        indexes.append(train_indexes[i][j])

# returns 100 indexes of 100 classes -> 10000 training images
X_train = X_train[indexes]
y_train = y_train[indexes]

print X_train.shape
print y_train.shape
"""
# one-hot encoding for 100 classes
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
nb_classes = len(y_test[1])

X_train_mean = np.mean(X_train, axis=0)
X_test_mean = np.mean(X_test, axis=0)

X_train -= X_train_mean
X_test -= X_test_mean

model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape=(32,32,3), activation='relu', border_mode = 'same'))
model.add(Cropping2D(cropping=((2,2),(2,2))))
#model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.5))
#model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
#model.add(MaxPooling2D(pool_size=(2,2)))
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


epochs = 30
l_rate = 0.01
decay = l_rate/epochs
sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Mirroring images 
datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, horizontal_flip=True)

datagen.fit(X_train)

hist = model.fit_generator(datagen.flow(X_train,y_train, batch_size=64),samples_per_epoch=len(X_train), nb_epoch=30, validation_data=(X_test, y_test))


#hist = model.fit(X_train, y_train, batch_size = 64, nb_epoch=5, validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1] * 100)



# plots of accuracy and loss
f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(hist.history['loss'])
ax1.plot(hist.history['val_loss'])
ax1.set_title('Loss Rate')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Training interations')
ax1.legend(['Training', 'Testing'], loc='upper left')
plt.show()
f1.savefig('Part5_Loss_rate1.png')

ax2 = f2.add_subplot(111)
ax2.plot(hist.history['val_acc'])
ax2.set_title('Accuracy Rate')
ax2.set_ylabel('Accuracy %')
ax2.set_xlabel('Training iterations')
ax2.legend(['Testing'], loc='upper left')
plt.show()
f2.savefig('Part5_Accuracy_plot1.png')


                                  

"""
# plots of accuracy and loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss rate with GCN')
plt.ylabel('loss')
plt.xlabel('training interations')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('lossmodel10.png')

# Classification accuracy on the test set vs. training iterations.
plt.plot(hist.history['val_acc'])
plt.title('Accuracy % with 1 extra hidden layer')
plt.ylabel('Accuracy %')
plt.xlabel('Training iterations')
plt.legend(['Testing'], loc='upper left')
plt.show()
plt.savefig('Extra_hidden_layer_acc1.png')
"""







