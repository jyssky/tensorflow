# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple image classification with Inception.

Run image classification with your model.

This script is usually used with retrain.py found in this same
directory.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.

It outputs human readable strings of the top 5 predictions along with
their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg

NOTE: To learn to use this file and retrain.py, please see:

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os.path

import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
  '--num_top_predictions',
  type=int,
  default=5,
  help='Display this many predictions.')
parser.add_argument(
  '--graph',
  required=True,
  type=str,
  help='Absolute path to graph file (.pb)')
parser.add_argument(
  '--output_layer',
  type=str,
  default='final_result:0',
  help='Name of the result operation')
parser.add_argument(
  '--input_layer',
  type=str,
  default='DecodeJpeg/contents:0',
  help='Name of the input operation')
parser.add_argument(
  '--csv',
  type=str,
  default='',
  help='output csv file name')
parser.add_argument(
  '--test_image_dir',
  type=str,
  default='',
  help='Directory for test images')
parser.add_argument(
  '--count_test_files',
  type=int,
  default=1531,
  help='Count of test images in test_image_dir')


def load_image(filename):
  """Read in the image_data to be classified."""
  filename = os.path.join(FLAGS.test_image_dir, str(filename) + '.jpg')
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, input_layer_name, output_layer_name):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    keep_prob = sess.graph.get_tensor_by_name('final_training_ops/dropout/keep_prob:0')
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data, keep_prob: 1.0})

    return predictions[0]


def main(argv):
  """Runs inference on an image."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  if not tf.gfile.Exists(FLAGS.test_image_dir):
    tf.logging.fatal('test images dir does not exist %s', FLAGS.image)

  # if not tf.gfile.Exists(FLAGS.labels):
  #     tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)

  output_df = pd.DataFrame(columns=['name', 'invasive'])

  for index in range(1, FLAGS.count_test_files + 1):
    # load image
    image_data = load_image(index)
    score = run_graph(image_data, FLAGS.input_layer, FLAGS.output_layer)
    output_df.loc[index] = [int(index), '%.9f' % float(score)]

  print(output_df)
  output_df.to_csv(FLAGS.csv, index=False)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1] + unparsed)
