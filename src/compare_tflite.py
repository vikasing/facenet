"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import align.detect_face
import imageio
from PIL import Image
from numpy import dot
from numpy.linalg import norm
import time
def main(args):
    interpreter = load_model('/home/vik/data/model/inference/fn_mobile.tflite')
    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    nrof_images = len(args.image_files)
    img_dict = {}
    print('Images:')
    for i in range(nrof_images):
        img_dict[i] = images[i, :]
        print('%1d: %s' % (i, args.image_files[i]))
    print('')

    # Print distance matrix
    print('Distance matrix')
    print('    ', end='')
    for i in range(nrof_images):
        print('    %1d     ' % i, end='')
    print('')
    t1 = time.time_ns()
    emb_dict = get_embeddings(img_dict, interpreter)
    for i, e1 in emb_dict.items():
        print('%1d  ' % i, end='')
        a = e1[0, :]
        for j, e2 in emb_dict.items():
            b = e2[0, :]
            # dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
            dist = dot(a, b) / (norm(a) * norm(b))
            print('  %1.4f  ' % dist, end='')
        print('')
    print((time.time_ns() - t1) / 1000000)

def get_embeddings(images_dict, interpreter):
    for i, image in images_dict.items():
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_data = np.zeros((1, 160, 160, 3))
        input_data[0, :, :, :] = image
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        images_dict[i] = interpreter.get_tensor(output_details[0]['index'])
    return images_dict


def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        try:
            img = imageio.imread(os.path.expanduser(image), pilmode='RGB')
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            if len(bounding_boxes) < 1:
                image_paths.remove(image)
                print("can't detect face, remove ", image)
                continue
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = np.array(Image.fromarray(cropped).resize((image_size, image_size), Image.BILINEAR))
            prewhitened = prewhiten(aligned)
            img_list.append(prewhitened)
        except Exception as e:
            print('Bad file:', image)
    images = np.stack(img_list)
    return images


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
