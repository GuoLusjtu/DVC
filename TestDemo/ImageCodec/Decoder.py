import pickle
import os
import imageio
from skimage.color import rgb2yuv
from skimage.color import rgb2ycbcr
import tensorflow as tf
from scipy.misc import imread
import numpy as np
from argparse import ArgumentParser


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def decoder(loadmodel, loadfolder):

    graph = load_graph(loadmodel)

    y_q = graph.get_tensor_by_name('import/quant_y:0')
    z_q = graph.get_tensor_by_name('import/quant_z:0')
    reconframe = graph.get_tensor_by_name('import/build_towers/tower_0/ICLR18/train_net/ReconFrame:0')

    with tf.Session(graph=graph) as sess:

        with open(loadfolder + 'quantized_y.pkl', 'rb') as f:
            y_val = pickle.load(f)

        with open(loadfolder + 'quantized_z.pkl', 'rb') as f:
            z_val = pickle.load(f)

        reconframe_val = sess.run([reconframe], feed_dict={y_q: y_val, z_q: z_val})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--DecoderModel', type=str, dest="loadmodel", default='./model.pb', help="encoder model")
    parser.add_argument('--loadpath', type=str, dest="loadfolder", default='pkl', help="load pkl folder")

    args = parser.parse_args()
    decoder(**vars(args))