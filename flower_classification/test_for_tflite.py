import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import PIL
from PIL import Image
import numpy as np
import time

# 加载TFLite模型并分配张量

img_row, img_column = 180, 180
input_mean = 0.0
input_std = 1.0

path_1 = r"./model.tflite"
labels_path ="./label.txt"

def load_labels(filename):
    my_labels = []
    input_file = open(filename, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels

interpreter = tf.lite.Interpreter(path_1)
interpreter.allocate_tensors()

# obtaining the input-output shapes and types
input_details = interpreter.get_input_details() # 输入
output_details = interpreter.get_output_details()  # 输出
print(input_details, output_details)

input_shape = input_details[0]['shape']  # 获取输入的shape
print(input_shape)
#img_row = input_shape[1]
#img_column = input_shape[2]

# file selection window for input selection
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
input_img = Image.open(file_path)
input_img = input_img.resize((img_row, img_column))
input_img = np.expand_dims(input_img, axis=0)
input_img = (np.float32(input_img) - input_mean) / input_std

interpreter.set_tensor(input_details[0]['index'], input_img) # 输入给模型

# 执行推理
interpreter.invoke()

# 取出结果 + 后处理
output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)

top_k = results.argsort()[-5:][::-1]
labels = load_labels(labels_path)
for i in top_k:
    print('{0:08.6f}'.format(float(results[i] / 255.0)) +":", labels[i])
