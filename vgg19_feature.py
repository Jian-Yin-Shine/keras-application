# vgg19从中间层抽取特征
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

import numpy as np

base_model = VGG19(weights='imagenet')
base_model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'timg.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_feature = model.predict(x)

