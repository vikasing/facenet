
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def main():
    converter = tf.lite.TFLiteConverter.from_saved_model('/home/vik/data/20200814-final')
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)