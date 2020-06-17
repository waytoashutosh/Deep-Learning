import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir



def train_happy_sad_model():
    
    
    DESIRED_ACCURACY = 0.999
    class myCallback(tf.keras.callbacks.Callback):
     
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>=DESIRED_ACCURACY):
              print("\nReached 99.9% accuracy so cancelling training!")
              self.model.stop_training = True

        
  

    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
     
        tf.keras.layers.Conv2D(64, (3,3),input_shape=(150,150,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=300, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
        
        
    ])

    from tensorflow.keras.optimizers import RMSprop
	model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
     
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255)

   
    train_generator = train_datagen.flow_from_directory(
 
        '/tmp/h-or-s',
        target_size=(150,150),
        batch_size=10,
        class_mode='binary'
        
    )
  
    
    history = model.fit_generator(
        
        train_generator,
        steps_per_epoch=8,
        epochs=20,
        callbacks=[callbacks]
    )
   
    return history.history['acc'][-1]

train_happy_sad_model()



get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')

