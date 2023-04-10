import tensorflow as tf
import numpy as np

def intensity_loss(gen_frames, gt_frames, l_num):
    return tf.reduce_mean(tf.abs((gen_frames - gt_frames) ** l_num))

def gradient_loss(gen_frames, gt_frames, alpha):
    channels = gen_frames.get_shape().as_list()[-1]
    pos = tf.constant(np.identity(channels), dtype=tf.float32)     # 3 x 3
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return tf.reduce_mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

def flow_loss(genrated, groud_truth):
    return tf.reduce_mean(tf.abs(genrated['flow'] - groud_truth['flow']))