import tensorflow as tf

saver = tf.train.import_meta_graph('/media/covisgpu5/D/models/research/slim/just/model.ckpt-0.meta')
sess = tf.Session()
saver.restore(sess, '/media/covisgpu5/D/models/research/slim/just/model.ckpt-0')
graph = sess.graph
print([node.name for node in graph.as_graph_def().node])