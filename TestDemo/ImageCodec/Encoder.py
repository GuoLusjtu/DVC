import pickle
import os
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


def encoder(loadmodel, input_path, outputfolder):
    graph = load_graph(loadmodel)

    inputImage = graph.get_tensor_by_name('import/input_image:0')
    z_feature = graph.get_tensor_by_name('import/build_towers/tower_0/ICLR18/train_net/z_fearure:0')
    reconframe = graph.get_tensor_by_name('import/build_towers/tower_0/ICLR18/train_net/ReconFrame:0')
    y_feature = graph.get_tensor_by_name('import/build_towers/tower_0/ICLR18/train_net/y_feature:0')
    psnr = graph.get_tensor_by_name('import/build_towers/tower_0/ICLR18/train_net/distortion/PSNR:0')
    bpp = graph.get_tensor_by_name('import/build_towers/tower_0/ICLR18/train_net/rate/bits_est:0')

    with tf.Session(graph=graph) as sess:

        im1 = imread(input_path)
        im1 = im1 / 255.0
        im1 = np.expand_dims(im1, axis=0)

        bpp_est, reconframe_val, y_fearure_val, z_feature_val, psnr_val = sess.run(
            [bpp, reconframe, y_feature, z_feature, psnr], feed_dict={inputImage: im1})
        print(bpp_est)
        print(psnr_val)

        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

        output = open(outputfolder + 'quantized_y.pkl', 'wb')
        pickle.dump(y_fearure_val, output)

        output = open(outputfolder + 'quantized_z.pkl', 'wb')
        pickle.dump(z_feature_val, output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--EncoderModel', type=str, dest="loadmodel", default='./model.pb', help="encoder model")
    parser.add_argument('--input_frame', type=str, dest="input_path", default='./im001.png', help="input image path")
    parser.add_argument('--outputpath', type=str, dest="outputfolder", default='pkl', help="output pkl folder")

    args = parser.parse_args()
    encoder(**vars(args))