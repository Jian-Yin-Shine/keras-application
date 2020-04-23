import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
import numpy as np

model = ResNet50(weights='imagenet')
img_path = 'timg.jpeg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print(x.shape)

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

model.summary()


