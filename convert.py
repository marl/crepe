import sys
import tensorflowjs as tfjs
import keras
from keras.layers import *
from keras.models import Model


def crepe(optimizer, model_capacity=32):
    layers = [1, 2, 3, 4, 5, 6]
    filters = [n * model_capacity for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), name='crepe_input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='crepe_input_reshape')(x)

    for layer, filters, width, strides in zip(layers, filters, widths, strides):
        y = Conv2D(filters, (width, 1), strides=strides, padding='same',
                   activation='relu', name="crepe_conv%d" % layer)(y)
        y = BatchNormalization(name="crepe_conv%d_BN" % layer)(y)
        y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid',
                         name="crepe_conv%d_maxpool" % layer)(y)
        y = Dropout(0.25, name="crepe_conv%d_dropout" % layer)(y)

    y = Flatten(name="crepe_flatten")(y)
    y = Dense(360, activation='sigmoid', name="crepe_classifier")(y)

    model = Model(inputs=x, outputs=y)
    model.compile(optimizer, 'binary_crossentropy')

    return model

print("Loading the original model ...")
original = keras.models.load_model(sys.argv[1])

print("Constructing a new model for tfjs ...")
model = crepe('adam', 4)

print("Copying weights ...")
for i in range(26):
    model.layers[i].set_weights(original.layers[i].get_weights())
model.layers[-1].set_weights(original.layers[-1].get_weights())

print("Saving tfjs model ...")
tfjs.converters.save_keras_model(model, 'model')

print("Done.")

