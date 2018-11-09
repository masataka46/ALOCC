import numpy as np
# import os
import tensorflow as tf
# from PIL import Image
# import utility as Utility
# import argparse

class ALOCC():
    def __init__(self, img_channel, noise_mean, noise_stddev, seed, base_channel, keep_prob):
        self.IMG_CHANNEL = img_channel  # 3
        self.SEED = seed
        np.random.seed(seed=self.SEED)
        self.BASE_CHANNEL = base_channel  # 64
        self.KEEP_PROB = keep_prob
        self.NOISE_MEAN = noise_mean
        self.NOISE_STDDEV = noise_stddev

    def leaky_relu(self, x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def gaussian_noise(self, input, std):  # used at discriminator
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32, seed=self.SEED)
        return input + noise

    def conv2d(self, input, in_channel, out_channel, k_size, stride, seed):
        w = tf.get_variable('w', [k_size, k_size, in_channel, out_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding="SAME", name='conv') + b
        return conv

    def conv2d_transpose(self, input, in_channel, out_channel, k_size, stride, seed):
        w = tf.get_variable('w', [k_size, k_size, out_channel, in_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        out_shape = tf.stack(
            [tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, tf.constant(out_channel)])
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=out_shape, strides=[1, stride, stride, 1],
                                        padding="SAME") + b
        return deconv

    def conv2d_transpose_with_shape(self, input, in_channel, out_channel, k_size, stride, shape, seed):
        w = tf.get_variable('w', [k_size, k_size, out_channel, in_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=shape, strides=[1, stride, stride, 1],
                                        padding="SAME") + b
        return deconv


    def batch_norm(self, input):
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        scale = tf.get_variable('scale', [n_out], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0))
        batch_mean, batch_var = tf.nn.moments(input, [0])
        bn = tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.0001, name='batch_norm')
        return bn

    def fully_connect(self, input, in_num, out_num, seed):
        w = tf.get_variable('w', [in_num, out_num], initializer=tf.random_normal_initializer
        (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_num], initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input, w, name='fc') + b
        return fc

    def encoder(self, x, reuse=False):  # x is expected [n, 28, 28, 3]
        with tf.variable_scope('encoder', reuse=reuse):
            with tf.variable_scope("add_noise"):
                noise = tf.random_normal(tf.shape(x), dtype=tf.float32, mean=self.NOISE_MEAN, stddev=self.NOISE_STDDEV)
                x_noise = x + noise

            with tf.variable_scope("layer1"):  # layer1 conv nx28x28x3 -> nx14x14x64
                conv1 = self.conv2d(x_noise, self.IMG_CHANNEL, self.BASE_CHANNEL, 5, 2, self.SEED)
                bn1 = self.batch_norm(conv1)
                self.e_lr1 = self.leaky_relu(bn1, alpha=0.1)

            with tf.variable_scope("layer2"):  # layer2 conv nx14x14x64 -> nx7x7x128
                conv2 = self.conv2d(self.e_lr1, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, 5, 2, self.SEED)
                bn2 = self.batch_norm(conv2)
                self.e_lr2 = self.leaky_relu(bn2, alpha=0.1)

            with tf.variable_scope("layer3"):  # layer3 conv nx7x7x128 -> nx4x4x256
                conv3 = self.conv2d(self.e_lr2, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, 5, 2, self.SEED)
                bn3 = self.batch_norm(conv3)
                self.e_lr3 = self.leaky_relu(bn3, alpha=0.1)

            with tf.variable_scope("layer4"):  # layer4 conv nx4x4x256 -> nx2x2x512
                conv4 = self.conv2d(self.e_lr3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 8, 5, 2, self.SEED)
                bn4 = self.batch_norm(conv4)
                self.e_lr4 = self.leaky_relu(bn4, alpha=0.1)

        return self.e_lr4

    def decoder(self, x, reuse=False):  # z is expected [n, 200]
        with tf.variable_scope('decoder', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer1 deconv nx2x2x512 -> nx4x4x256
                shape1 = tf.shape(self.e_lr3)
                deconv1 = self.conv2d_transpose_with_shape(x, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 4, 5, 2, shape1, self.SEED)
                bn1 = self.batch_norm(deconv1)
                rl1 = tf.nn.relu(bn1)

            with tf.variable_scope("layer2"):  # layer2 deconv nx4x4x256 -> nx7x7x128
                shape2 = tf.shape(self.e_lr2)
                deconv2 = self.conv2d_transpose_with_shape(rl1, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 2, 5, 2, shape2, self.SEED)
                bn2 = self.batch_norm(deconv2)
                rl2 = tf.nn.relu(bn2)

            with tf.variable_scope("layer3"):  # layer2 deconv nx7x7x128 -> nx14x14x64
                shape3 = tf.shape(self.e_lr1)
                deconv3 = self.conv2d_transpose_with_shape(rl2, self.BASE_CHANNEL * 2, self.BASE_CHANNEL, 4, 2, shape3, self.SEED)
                bn3 = self.batch_norm(deconv3)
                rl3 = tf.nn.relu(bn3)

            with tf.variable_scope("layer4"):  # layer3 deconv nx14x14x64 -> nx28x28x3
                deconv4 = self.conv2d_transpose(rl3, self.BASE_CHANNEL, self.IMG_CHANNEL, 4, 2, self.SEED)
                tanh4 = tf.tanh(deconv4)

        return tanh4

    def discriminator(self, x, reuse=False, is_training=True):  # z[n, 200], x[n, 28, 28, 1]
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer1 conv nx28x28x3 -> nx14x14x64
                conv1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL, 5, 2, self.SEED)
                bn1 = self.batch_norm(conv1)
                lr1 = self.leaky_relu(bn1, alpha=0.1)

            with tf.variable_scope("layer2"):  # layer2 conv nx14x14x64 -> nx7x7x128
                conv2 = self.conv2d(lr1, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, 5, 2, self.SEED)
                bn2 = self.batch_norm(conv2)
                lr2 = self.leaky_relu(bn2, alpha=0.1)

            with tf.variable_scope("layer3"):  # layer3 conv nx7x7x128 -> nx4x4x256
                conv3 = self.conv2d(lr2, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, 5, 2, self.SEED)
                bn3 = self.batch_norm(conv3)
                lr3 = self.leaky_relu(bn3, alpha=0.1)

            with tf.variable_scope("layer4"):  # layer4 conv nx4x4x256 -> nx2x2x512
                conv4 = self.conv2d(lr3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 8, 5, 2, self.SEED)
                bn4 = self.batch_norm(conv4)
                lr4 = self.leaky_relu(bn4, alpha=0.1)

            with tf.variable_scope("layer5"): # layer4 fc nx2048 -> nx1
                shape5 = tf.shape(lr4)
                reshape5 = tf.reshape(lr4, [shape5[0], shape5[1] * shape5[2] * shape5[3]])
                fc5 = self.fully_connect(reshape5, 2048, 1, self.SEED)
                self.logits = tf.nn.sigmoid(fc5)

        return self.logits

    def cross_entropy_loss_accuracy(self, prob, tar):
        crossEntropy_loss = - tf.reduce_mean(tf.multiply(tar, tf.log(tf.clip_by_value(prob, 1e-10, 1.0))) +
                                             tf.multiply(1.-tar, tf.log(tf.clip_by_value(1-prob, 1e-10, 1.0))),
                                        name='cross_entropy_loss')
        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(tar, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print('crossEntropy_loss.get_shape(), ', crossEntropy_loss.get_shape())
        return crossEntropy_loss, accuracy


