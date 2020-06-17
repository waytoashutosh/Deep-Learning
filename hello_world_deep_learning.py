#this is to create a simple housing price prediction where prices = 50k * rooms + 50k

import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
    xs = np.array([2.0, 12.0, 9.0, 10.0], dtype='float')
    ys = 0.5*xs+0.5
    model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs,ys,epochs=500)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)   #output = prediction * 100k
