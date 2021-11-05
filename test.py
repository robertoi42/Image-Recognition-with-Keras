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
from sklearn.metrics import classification_report, confusion_matrix

src_path_test = "data/test/"
num_of_test_samples = 105
batch_size = 1

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)
#Best
#Customsgd0.01augdrop4.h5
#Vgg16Adam0.0001augnodropout2.h5
#Resadam0.00001aug3.h5
#XceptionSGD0.01augdropthisone.h5

reconstructed_model = keras.models.load_model("Customsgd0.01augnodropthisone.h5")
'''
score = reconstructed_model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
Y_pred = reconstructed_model.predict_generator(test_generator, num_of_test_samples // batch_size)
y_pred = np.argmax(Y_pred, axis=1)

conf_matrix=confusion_matrix(test_generator.classes, y_pred)

target_names = ['2', '5', '10','20','50','100','200']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

ax.set_xticklabels([''] + target_names)
ax.set_yticklabels([''] + target_names) 
    
ax.yaxis.set_label_position("right")

plt.xlabel('Resultado', fontsize=18,labelpad=10)
plt.ylabel('        Real', fontsize=18,rotation=0)
plt.title('$\it{Custom}$ com $\it{data}$ $\it{augmentation}$ e sem $\it{Dropout}$', fontsize=18)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False)

plt.show()
