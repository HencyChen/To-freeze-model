import tensorflow as tf 

LOG_DIR = '/media/covisgpu5/D/models/research/slim/just/'
model_filename = '/media/covisgpu5/D/models/research/slim/just/forzen_graph.pb'

with tf.Session() as sess:
    with tf.gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
    writer = tf.summary.FileWriter(LOG_DIR, graph_def)
writer.close()
