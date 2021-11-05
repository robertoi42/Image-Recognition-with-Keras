from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications
import tensorflow as tf
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
src_path_train = "data2/train/"
src_path_test = "data2/test/"
src_path_valid = "data2/valid/"
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
	vgg16_model = applications.VGG16(input_shape=(224,224,3), weights="imagenet", include_top=False)
	model = Sequential()
	
	for layer in vgg16_model.layers:
		model.add(layer)

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu'))

	model.add(Dense(7, activation='softmax'))
	model.compile(loss="categorical_crossentropy",optimizer=tf.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])
	return model


model = prepare_model()
hist = model.fit(train_generator,
                    validation_data = valid_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = valid_generator.n//valid_generator.batch_size,
                    epochs=70,callbacks=[cb])

print(cb.logs)
print(sum(cb.logs))                    
          
score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

f = open('historyVgg16Adam0.0001noaugdroptest.pck', 'wb')
pickle.dump(hist.history, f)
f.close()

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='Vgg16Adam0.0001noaugdroptest.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
#retrieve
#f = open('historycustom.pckl', 'rb')
#history = pickle.load(f)
#f.close()

model.save("Vgg16Adam0.0001noaugdroptest.h5")
