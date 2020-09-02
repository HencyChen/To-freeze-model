import os
import sys

from google.protobuf import text_format

import tensorflow as tf
from tensorflow.python.platform import gfile


with tf.Session() as sess:
    model_filename = sys.argv[1]
    with gfile.FastGFile(model_filename, 'r') as f:
        graph_def = tf.GraphDef()
        text_format.Merge(f.read(), graph_def)
        g_in = tf.import_graph_def(graph_def)

LOGDIR = os.path.join(os.path.dirname(model_filename))
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
