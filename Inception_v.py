# 在 inception_v3上进行微调

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image

# 构建不带分类器的预训练模型
base_model = InceptionV3(weights='imagenet', include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

# 添加一个分类起，假设有200个类别
predictions = Dense(200, activation='softmax')(x)

# 完整模型
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# 首先我们锁住Inceptionv3的所有层
for layer in base_model.layers :
    layer.trainable = False

# 编译模型
model.compile(optimizer=tf.optimizers.RMSprop(0.01), loss=tf.keras.losses.categorical_crossentropy)

# 在新的训练集上训练
# model.fit(...)
# model.fit_generator(...)

# 在顶层应该训练好的情况下，开始微调inception v3的层
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for layer in model.layers[: 249]:
    layer.trainable = False
for layer in model.layers[249: ]:
    layer.trainable = True

# 从新编译模型，才能使得上面的修改生效
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
              loss=tf.keras.losses.categorical_crossentropy)

# 再次训练
# model.fit(...)
# model.fit_generator(...)

# 可以自定义输入张量作为输入
input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
model = InceptionV3(input_tensor = input_tensor, weigths='imagenet', include_top=False)