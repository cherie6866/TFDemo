import tensorflow as tf
import numpy as np

from PIL import Image

#分类名称，与训练的顺序保持一致即可
class_names = ['cloudy', 'rainy', 'snow', 'sunny']

img_height = 180
img_width = 180
model = tf.keras.models.load_model(r'./model_weather_classification.h5')
img = Image.open("./data/weather_classification/snow/snow_00023.jpg")
image = tf.image.resize(img, [img_height, img_width])

img_array = tf.expand_dims(image, 0)

predictions = model.predict(img_array)

print("预测结果为：",class_names[np.argmax(predictions)])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)





