import tensorflow as tf
import os,PIL,pathlib
import matplotlib.pyplot as plt
import numpy             as np
from tensorflow          import keras
from tensorflow.keras    import layers,models
from PIL import Image

data_dir = "./data/weather_photos"
print("图片路径：", data_dir)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print("图片总数为：",image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
class_names = train_ds.class_names
print(class_names)

plt.figure(figsize = (20, 10))

for images, labels in train_ds.take(1):
    for i in range(20):
        ax = plt.subplot(5, 10, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
#plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 4

model = models.Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255 , input_shape = (img_height ,
    img_width , 3)) ,
    
    layers.Conv2D(16,(3,3) , activation = 'relu' , input_shape = (img_height, img_width, 3)) ,  # 卷积层1，卷积核3*3
    layers.AveragePooling2D((2,2)) ,  # 池化层1，2*2采样
    layers.Conv2D(32,(3,3) , activation = 'relu') ,  # 卷积层2，卷积核3*3
    layers.AveragePooling2D((2,2)) ,  # 池化层2，2*2采样
    layers.Conv2D(64,(3,3) , activation = 'relu') ,  # 卷积层3，卷积核3*3
    layers.Dropout(0.3) , 
    layers.Flatten() ,  # Flatten层，连接卷积层与全连接层
    layers.Dense(128 , activation = 'relu') ,  # 全连接层，特征进一步提取
    layers.Dense(num_classes)  # 输出层，输出预期结果
])

model.summary()  # 打印网络结构

# 设置优化器
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt,
           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           metrics=['accuracy'])


epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

model.save(r'./model_weather_classification.h5')

#模型已经训练完成，可以立马用它来进行预测
#img = Image.open("./data/weather_classification/snow/snow_00004.jpg")
#image = tf.image.resize(img, [img_height, img_width])/255.0
#img_array = tf.expand_dims(image, 0)
#predictions = model.predict(img_array)
#print("预测结果为：",class_names[np.argmax(predictions)])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
