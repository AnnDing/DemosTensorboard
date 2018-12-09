"""
A demo to visualize the vectors of MNIST images in tensorboard.
We use the projector plugin in tensorboard.

1. create a variable A for the vectors to visualize
2. initialize the variable A, and save it
3. save metadata to record labels for each vectors in A

Howto:
python visual_mnist.py
tensorboard --logdir logs
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


LOG_DIR = 'logs'

# create log directory
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
# create metadata
metadata_path = 'metadata.tsv'

# get mnistdata
mnist = input_data.read_data_sets('MNIST_data')
images = tf.Variable(mnist.test.images, name='images')

# save metadata
with open(os.path.join(LOG_DIR, metadata_path), 'w') as metadata_file:
    for row in mnist.test.labels:
        metadata_file.write('%d\n' % row)

with tf.Session() as sess:
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # Vectors to be visualized
    sess.run(images.initializer)
    saver = tf.train.Saver([images])
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    # Projector config
    config = projector.ProjectorConfig()
    # Create a embedding object to be visualized
    embedding = config.embeddings.add()
    # Link the Tensor
    embedding.tensor_name = images.name
    # Link the metadata file
    embedding.metadata_path = metadata_path

    # Saves the wirter and config
    projector.visualize_embeddings(writer, config)