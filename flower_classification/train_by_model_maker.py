import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

image_path = "/home/wangzhili/.keras/datasets/flower_photos"

#第 1 步：加载特定于设备端 ML 应用的输入数据，并将其拆分为训练数据和测试数据。
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

#第 2 步：自定义 TensorFlow 模型。
model = image_classifier.create(train_data)

#第 3 步：评估模型。
loss, accuracy = model.evaluate(test_data)

#第 4 步：导出为 TensorFlow Lite 模型。
model.export(export_dir='.')
