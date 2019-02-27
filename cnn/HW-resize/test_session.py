import tensorflow as tf

for _ in range(3):
    with tf.Graph().as_default() as graph:
        var = tf.Variable(0)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print(len(graph._nodes_by_name.keys()))