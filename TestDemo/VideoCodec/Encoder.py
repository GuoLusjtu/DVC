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


def encoder(loadmodel, input_path, refer_path, outputfolder):
    graph = load_graph(loadmodel)

    Res = graph.get_tensor_by_name('import/build_towers/tower_0/DVC_Model/train_net/Residual_Feature:0')
    inputImage = graph.get_tensor_by_name('import/input_image:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')
    Res_prior = graph.get_tensor_by_name('import/build_towers/tower_0/DVC_Model/train_net/Residual_Prior_Feature:0')
    motion = graph.get_tensor_by_name('import/build_towers/tower_0/DVC_Model/train_net/Motion_Feature:0')
    bpp = graph.get_tensor_by_name('import/build_towers/tower_0/DVC_Model/train_net/rate/Estimated_Bpp:0')
    psnr = graph.get_tensor_by_name('import/build_towers/tower_0/DVC_Model/train_net/distortion/PSNR:0')
    reconframe = graph.get_tensor_by_name('import/build_towers/tower_0/DVC_Model/train_net/ReconFrame:0')
    mode = graph.get_tensor_by_name('import/Mode:0')

    with tf.Session(graph=graph) as sess:

        im1 = imread(input_path)
        im2 = imread(refer_path)
        im1 = im1 / 255.0
        im2 = im2 / 255.0
        im1 = np.expand_dims(im1, axis=0)
        im2 = np.expand_dims(im2, axis=0)

        bpp_est, Res_q, Res_prior_q, motion_q, psnr_val = sess.run([bpp, Res, Res_prior, motion, psnr],
                                                                   feed_dict={
                                                                       inputImage: im2,
                                                                       previousImage: im1,
                                                                       mode: 1
                                                                   })
    print(bpp_est)
    print(psnr_val)
    if not os.path.exists(outputfolder):
        os.mkdir(ou)

    output = open(outputfolder + 'quantized_res_feature.pkl', 'wb')
    pickle.dump(Res_q, output)

    output = open(outputfolder + 'quantized_res_prior_feature.pkl', 'wb')
    pickle.dump(Res_prior_q, output)

    output = open(outputfolder + 'quantized_motion_feature.pkl', 'wb')
    pickle.dump(motion_q, output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--EncoderModel', type=str, dest="loadmodel", default='./model.pb', help="encoder model")
    parser.add_argument('--input_frame', type=str, dest="input_path", default='./im001.png', help="input image path")
    parser.add_argument('--refer_frame', type=str, dest="refer_path", default='./im002.png', help="refer image path")
    parser.add_argument('--outputpath', type=str, dest="outputfolder", default='pkl', help="output pkl folder")

    args = parser.parse_args()
    encoder(**vars(args))