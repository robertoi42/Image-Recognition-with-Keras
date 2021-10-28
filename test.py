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

src_path_test = "data2/test/"

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42
)

reconstructed_model = keras.models.load_model("Vgg16Adam0.0001noaugdropout2.h5")

score = reconstructed_model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
