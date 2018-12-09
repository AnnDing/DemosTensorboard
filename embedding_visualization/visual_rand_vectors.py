import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

LOG_DIR = 'logs'

# create log directory
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# create metadata
metadata_path = 'labels.tsv'

# create 100 vectors with dimension size 7
embedding_var = tf.Variable(tf.truncated_normal([100, 7]), name='embedding')

with open(os.path.join(LOG_DIR, metadata_path), 'w') as metadata_file:
    labels = np.random.randint(3, size=100)
    for l in labels:
        metadata_file.write('%d\n' % l)

with tf.Session() as sess:
    # Create summary writer.
    writer = tf.summary.FileWriter('./logs', sess.graph)

    # Initialize embedding_var
    sess.run(embedding_var.initializer)
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, os.path.join(LOG_DIR, 'embedding.ckpt'))

    # Create Projector config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = metadata_path
    projector.visualize_embeddings(writer, config)
writer.close()