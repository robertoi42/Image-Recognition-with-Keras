from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import pickle
from timeit import default_timer as timer
    
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()       
src_path_train = "data/train/"
src_path_test = "data/test/"
src_path_valid = "data/valid/"
num_classes = 7
'''
train_datagen = ImageDataGenerator(
        rescale=1 / 255.0)
'''     
 
train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=45,
        zoom_range=[0.8,1.2],
        width_shift_range=50,
        height_shift_range=50,
        brightness_range=[0.8,1.2],
        shear_range=30,
        horizontal_flip=True,
        fill_mode="nearest")          

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

batch_size = 16
train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)

valid_generator = test_datagen.flow_from_directory(
    directory=src_path_valid,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42
)

def prepare_model():
	model = Sequential()
	model.add(Conv2D(16, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(224,224,3)))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D((2, 2),padding='same'))
	model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D((2, 2),padding='same'))
	model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))                  
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='linear'))
	model.add(LeakyReLU(alpha=0.1))           
	model.add(Dropout(0.3))
	model.add(Dense(64, activation='relu'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss="categorical_crossentropy",optimizer=tf.optimizers.SGD(learning_rate=0.01),metrics=['accuracy'])
	return model

model = prepare_model()
hist = model.fit(train_generator,
                    validation_data = valid_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    epochs=100, callbacks=[cb])

print(cb.logs)
print(sum(cb.logs))                    
                       
score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='Customsgd0.01augdrop4.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
f = open('historycustomsgd0.01augdrop4.pck', 'wb')
pickle.dump(hist.history, f)
f.close()

#retrieve
#f = open('historycustom.pckl', 'rb')
#history = pickle.load(f)
#f.close()

model.save("Customsgd0.01augdrop4.h5")


