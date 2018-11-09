import numpy as np
import os
import tensorflow as tf
import utility as Utility
import argparse
from model import ALOCC as Model
from make_datasets_MNIST import Make_datasets_MNIST as Make_datasets

def parser():
    parser = argparse.ArgumentParser(description='train LSGAN')
    parser.add_argument('--batch_size', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='log180926', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch')
    parser.add_argument('--file_name', '-fn', type=str, default='./mnist.npz', help='file name of data')
    parser.add_argument('--valid_span', '-vs', type=int, default=5, help='validation span')
    return parser.parse_args()

args = parser()

#global variants
BATCH_SIZE = args.batch_size
LOGFILE_NAME = args.log_file_name
EPOCH = args.epoch
FILE_NAME = args.file_name
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNEL = 3
BASE_CHANNEL = 64
NOISE_MEAN = 0.0
NOISE_STDDEV = 0.3
TEST_DATA_SAMPLE = 5 * 5
L2_NORM = 0.001
KEEP_PROB_RATE = 0.5
SEED = 1234
VALID_SPAN = args.valid_span
np.random.seed(seed=SEED)
BOARD_DIR_NAME = './tensorboard/' + LOGFILE_NAME
OUT_IMG_DIR = './out_images_ALOCC' #output image file
out_model_dir = './out_models_ALOCC' #output model file
CYCLE_LAMBDA = 1.0
INLIER_NUM = 1.0
THRESHOLD_TAU = 0.5

try:
    os.mkdir(OUT_IMG_DIR)
    os.mkdir(out_model_dir)
    os.mkdir('log')
    os.mkdir('out_graph')
    os.mkdir('./out_images_Debug') #for debug
except:
    pass

make_datasets = Make_datasets(FILE_NAME, IMG_WIDTH, IMG_HEIGHT, SEED)
model = Model(IMG_CHANNEL, NOISE_MEAN, NOISE_STDDEV, SEED, BASE_CHANNEL, KEEP_PROB_RATE)

x_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL], name='x_') #image to classifier
d_dis_f_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_f_') #target of discriminator related to generator
d_dis_r_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_r_') #target of discriminator related to real image
is_training_ = tf.placeholder(tf.bool, name = 'is_training')

with tf.variable_scope('encoder_model'):
    enc = model.encoder(x_, reuse=False)

with tf.variable_scope('decoder_model'):
    x_dec = model.decoder(enc, reuse=False)

with tf.variable_scope('discriminator_model'):
    #stream around discriminator
    logits_r = model.discriminator(x_, reuse=False, is_training=is_training_) #real
    logits_f = model.discriminator(x_dec, reuse=True, is_training=is_training_) #fake

with tf.name_scope("loss"):
    loss_RD_r, accu_RD_r = model.cross_entropy_loss_accuracy(logits_r, d_dis_r_)
    loss_RD_f, accu_RD_f = model.cross_entropy_loss_accuracy(logits_f, d_dis_f_)
    loss_R = tf.reduce_mean(tf.square(x_dec - x_), name='Loss_R') #loss related to real image
    #total loss
    loss_D_total = loss_RD_f + loss_RD_r
    loss_R_total = loss_RD_f + CYCLE_LAMBDA * loss_R

tf.summary.scalar('loss_dis_total', loss_D_total)

merged = tf.summary.merge_all()

# t_vars = tf.trainable_variables()
enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

with tf.name_scope("train"):
    train_dis = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(loss_D_total, var_list=dis_vars
                                                                                , name='Adam_D')
    train_r = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5).minimize(loss_R_total, var_list=enc_vars+dec_vars
                                                                                , name='Adam_R')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(BOARD_DIR_NAME, sess.graph)

log_list = []
log_list.append(['epoch', 'AUC OCC1', 'AUC OCC2'])
#training loop
for epoch in range(0, EPOCH):
    sum_loss_dis_f = np.float32(0)
    sum_loss_dis_r = np.float32(0)
    sum_loss_dis_total = np.float32(0)
    sum_loss_r_total = np.float32(0)
    sum_loss_RD_f = np.float32(0)
    sum_loss_R = np.float32(0)
    len_data = make_datasets.make_data_for_1_epoch()

    for i in range(0, len_data, BATCH_SIZE):
        img_batch = make_datasets.get_data_for_1_batch(i, BATCH_SIZE)
        tar_g_1 = make_datasets.make_target_1_0(1.0, len(img_batch)) #1 -> real
        tar_g_0 = make_datasets.make_target_1_0(0.0, len(img_batch)) #0 -> fake

        #train discriminator
        sess.run(train_dis, feed_dict={x_: img_batch, d_dis_f_: tar_g_0, d_dis_r_: tar_g_1, is_training_:True})
        #train r-network
        sess.run(train_r, feed_dict={x_:img_batch, d_dis_f_: tar_g_1, is_training_:True})
        # loss for discriminator
        loss_dis_total_, loss_dis_r_, loss_dis_f_ = sess.run([loss_D_total, loss_RD_r, loss_RD_f],
                                                             feed_dict={x_: img_batch, d_dis_f_: tar_g_0,
                                                                        d_dis_r_: tar_g_1, is_training_:True})
        #loss for r-network
        loss_r_total_, loss_RD_f_, loss_R_ = sess.run([loss_R_total, loss_RD_f, loss_R], feed_dict={x_:img_batch,
                                                                            d_dis_f_: tar_g_1, is_training_:True})
        #for tensorboard
        merged_ = sess.run(merged, feed_dict={x_: img_batch, d_dis_f_: tar_g_0, d_dis_r_: tar_g_1, is_training_:True})

        summary_writer.add_summary(merged_, epoch)

        sum_loss_dis_f += loss_dis_f_ * len(img_batch)
        sum_loss_dis_r += loss_dis_r_ * len(img_batch)
        sum_loss_dis_total += loss_dis_total_ * len(img_batch)
        sum_loss_r_total += loss_r_total_ * len(img_batch)
        sum_loss_RD_f += loss_RD_f_ * len(img_batch)
        sum_loss_R += loss_R_ * len(img_batch)

    print("----------------------------------------------------------------------")
    print("epoch = {:}, R-Network Total Loss = {:.4f}, Discriminator Total Loss = {:.4f}".format(
        epoch, sum_loss_r_total / len_data, sum_loss_dis_total / len_data))
    print("Discriminator Real Loss = {:.4f}, Discriminator Fake Loss = {:.4f}".format(
        sum_loss_dis_r / len_data, sum_loss_dis_f / len_data))
    print("R-Network adv Loss = {:.4f}, R-Network Reconstruction Loss = {:.4f}".format(
        sum_loss_RD_f / len_data, sum_loss_R / len_data))

    if epoch % VALID_SPAN == 0:
        # score_A_list = []
        score_A_np = np.zeros((0, 3), dtype=np.float32)
        val_data_num = len(make_datasets.valid_all)
        for i in range(0, val_data_num, BATCH_SIZE):
            img_batch, tars_batch = make_datasets.get_valid_data_for_1_batch(i, BATCH_SIZE)
            logits_r_, logits_f_ = sess.run([logits_r, logits_f], feed_dict={x_:img_batch, is_training_:False})
            logits_r_re = np.reshape(logits_r_, (-1, 1))
            logits_f_re = np.reshape(logits_f_, (-1, 1))
            tars_batch_re = np.reshape(tars_batch, (-1, 1))
            score_A_np_tmp = np.concatenate((logits_r_re, logits_f_re, tars_batch_re), axis=1)
            score_A_np = np.concatenate((score_A_np, score_A_np_tmp), axis=0)

        # tp, fp, tn, fn, precision, recall = Utility.compute_precision_recall(score_A_np, INLIER_NUM)
        tp0, fp0, tn0, fn0, precision0, recall0, tp1, fp1, tn1, fn1, precision1, recall1 = \
                Utility.compute_precision_recall(score_A_np, INLIER_NUM, THRESHOLD_TAU)

        auc0, auc1 = Utility.make_ROC_graph(score_A_np, 'out_graph/' + LOGFILE_NAME, epoch, THRESHOLD_TAU)
        print("by OCC1, tp:{}, fp:{}, tn:{}, fn:{}, precision:{:.4f}, recall:{:.4f}, AUC:{:.4f}".format(tp0, fp0, tn0, fn0, precision0, recall0, auc0))
        print("by OCC2, tp:{}, fp:{}, tn:{}, fn:{}, precision:{:.4f}, recall:{:.4f}, AUC:{:.4f}".format(tp1, fp1, tn1, fn1, precision1, recall1, auc1))

        log_list.append([epoch, auc0, auc1])

        img_batch_in = make_datasets.get_data_for_1_batch(0, 10)
        img_batch_out, _ = make_datasets.get_valid_data_for_1_batch(0, 10)
        x_dec_in = sess.run(x_dec, feed_dict={x_:img_batch_in, is_training_:False})
        x_dec_out = sess.run(x_dec, feed_dict={x_:img_batch_out, is_training_:False})

        Utility.make_output_img(img_batch_out, img_batch_in, x_dec_out, x_dec_in, epoch, LOGFILE_NAME, OUT_IMG_DIR)


    #after learning
    Utility.save_list_to_csv(log_list, 'log/' + LOGFILE_NAME + '_auc.csv')

