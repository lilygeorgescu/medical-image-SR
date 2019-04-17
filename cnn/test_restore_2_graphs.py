import tensorflow as tf



with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read()) 

session_h_w = tf.Session(graph=graph_def)

        