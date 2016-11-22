from scipy.misc import imsave
from keras.datasets import cifar100
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import Tkinter as tk
from keras import backend as K


# Import Training and Testing data set 
# 50,000 32x32 training images over 100 labels
# 10,000 32x32 testing images
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#X_train = X_train / 255.0
#X_test = X_test / 255.0

# One hot encode the outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3),init = 'uniform', border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.4))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',W_constraint=maxnorm(3)))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',W_constraint=maxnorm(3)))
model.add(Dropout(0.4))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',W_constraint=maxnorm(3)))
model.add(Dropout(0.4))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',W_constraint=maxnorm(3)))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.7))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
print(model.summary())

epochs = 1
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit model
"""
# Fit model with image 
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    )

# compute quantities (std dev, mean, principal components) for feature normalization
datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train,y_train),samples_per_epoch=len(X_train), nb_epoch=25)

"""

# Evaluate model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print "Accuracy: %.2f%%" % (scores[1] * 100)
model.fit(X_train, y_train, batch_size=64, nb_epoch=epochs, validation_data=(X_test, y_test))
#print(hist.history.keys())

# pull out the weights from the evaluated models
layer_dict = dict([(layer.name, layer) for layer in model.layers])
weights_conv_1 = layer_dict['convolution2d_1'].get_weights()
weights_conv_2 = layer_dict['convolution2d_2'].get_weights()

# create a new model 
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3),  activation ='relu', border_mode = 'same', W_constraint=maxnorm(3)))
model.add(Convolution2D(32, 3, 3, activation ='relu', border_mode = 'same', W_constraint=maxnorm(3)))


layer_dict = dict([(layer.name, layer) for layer in model.layers])

# substitue in the pretrained weights
print model.summary()

layer_dict['convolution2d_7'].set_weights(weights_conv_1)
layer_dict['convolution2d_8'].set_weights(weights_conv_2)

layer_name = 'convolution2d_7'
filter_index = 1

first_layer = model.layers[-1]
input_img = first_layer.input

for filter_index in xrange(32):
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some noise
    img_width = 32
    img_height = 32
    input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
    step = 1
    # run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    img = input_img_data[0]
    img = deprocess_image(img)
    imsave('images/features/%s_filter_%d.png' % (layer_name, filter_index), img)
    break
