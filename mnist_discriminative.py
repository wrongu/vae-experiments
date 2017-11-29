import os
from models import discriminative_mnist
from keras.utils import to_categorical

###################
# LOAD MNIST DATA #
###################

from my_mnist import x_train, x_test, y_train, y_test
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

img_rows, img_cols = 28, 28
img_pixels = img_rows * img_cols

#################
# CREATE MODELS #
#################

ff_model = discriminative_mnist()
ff_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

###########
# FITTING #
###########

# Load pre-trained weights if they exist
weights_file = os.path.join("models", "mnist_discriminative.h5")
if os.path.exists(weights_file):
    ff_model.load_weights(weights_file)
else:
    ff_model.fit(x_train, y_train, shuffle=True, epochs=10, batch_size=100,
                 validation_data=(x_test, y_test))
    # Save trained model to a file
    ff_model.save_weights(weights_file)
