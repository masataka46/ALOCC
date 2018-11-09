import numpy as np
# import os
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import csv


def compute_precision_recall(score_A_np, inlier_num, threshold_tau):
    # print("score_A_np.shape, ", score_A_np.shape)
    # print("score_A_np[0,2], ", score_A_np[0,2])

    array_in = np.where(score_A_np[:, 2] == inlier_num)
    array_out = np.where(score_A_np[:, 2] != inlier_num)
    # print("len(array_in), ", len(array_in))
    # print("len(array_out), ", len(array_out))
    # mean_in = np.mean((score_A_np[array_in])[:, 0])
    # mean_out = np.mean((score_A_np[array_out])[:, 0])
    # medium = (mean_in + mean_out) / 2.0
    # print("mean_in, ", mean_in)
    # print("mean_out, ", mean_out)
    # print("medium, ", medium)
    array_upper0 = score_A_np[:, 0] >= threshold_tau
    array_lower0 = score_A_np[:, 0] < threshold_tau
    array_upper1 = score_A_np[:, 1] >= threshold_tau
    array_lower1 = score_A_np[:, 1] < threshold_tau
    # print("np.sum(array_upper0.astype(np.float32)), ", np.sum(array_upper0.astype(np.float32)))
    # print("np.sum(array_lower0.astype(np.float32)), ", np.sum(array_lower0.astype(np.float32)))
    # print("np.sum(array_upper1.astype(np.float32)), ", np.sum(array_upper1.astype(np.float32)))
    # print("np.sum(array_lower1.astype(np.float32)), ", np.sum(array_lower1.astype(np.float32)))
    array_in_tf = score_A_np[:, 2] == inlier_num
    array_out_tf = score_A_np[:, 2] == inlier_num
    # print("np.sum(array_in_tf.astype(np.float32)), ", np.sum(array_in_tf.astype(np.float32)))
    # print("np.sum(array_out_tf.astype(np.float32)), ", np.sum(array_out_tf.astype(np.float32)))

    tn0 = np.sum(np.equal(array_lower0, array_in_tf).astype(np.int32))
    tp0 = np.sum(np.equal(array_upper0, array_out_tf).astype(np.int32))
    fp0 = np.sum(np.equal(array_upper0, array_in_tf).astype(np.int32))
    fn0 = np.sum(np.equal(array_lower0, array_out_tf).astype(np.int32))
    precision0 = tp0 / (tp0 + fp0 + 0.00001)
    recall0 = tp0 / (tp0 + fn0 + 0.00001)
    
    tn1 = np.sum(np.equal(array_lower1, array_in_tf).astype(np.int32))
    tp1 = np.sum(np.equal(array_upper1, array_out_tf).astype(np.int32))
    fp1 = np.sum(np.equal(array_upper1, array_in_tf).astype(np.int32))
    fn1 = np.sum(np.equal(array_lower1, array_out_tf).astype(np.int32))
    precision1 = tp1 / (tp1 + fp1 + 0.00001)
    recall1 = tp1 / (tp1 + fn1 + 0.00001)

    return tp0, fp0, tn0, fn0, precision0, recall0, tp1, fp1, tn1, fn1, precision1, recall1


def save_graph(x, y, filename, epoch):
    plt.plot(x, y)
    plt.title('ROC curve ' + filename + ' epoch:' + str(epoch))
    # x axis label
    plt.xlabel("FP / (FP + TN)")
    # y axis label
    plt.ylabel("TP / (TP + FN)")
    # save
    plt.savefig(filename + '_ROC_curve_epoch' + str(epoch) +'.png')
    plt.close()


def make_ROC_graph(score_A_np, filename, epoch, threshold_tau):
    argsort_F = np.argsort(score_A_np, axis=0)[:, 0]
    score_A_np_sort_F = score_A_np[argsort_F][::-1]
    value_1_0_F = (np.where(score_A_np_sort_F[:, 2] == threshold_tau, 1., 0.)).astype(np.float32)
    # score_A_np_sort_0_1 = np.concatenate((score_A_np_sort, value_1_0), axis=1)
    sum_1_F = np.sum(value_1_0_F)
    len_s_F = len(score_A_np)
    sum_0_F = len_s_F - sum_1_F
    tp_F = np.cumsum(value_1_0_F).astype(np.float32)
    index_F = np.arange(1, len_s_F + 1, 1).astype(np.float32)
    fp_F = index_F - tp_F
    fn_F = sum_1_F - tp_F
    tn_F = sum_0_F - fp_F
    tp_ratio_F = tp_F / (tp_F + fn_F + 0.00001)
    fp_ratio_F = fp_F / (fp_F + tn_F + 0.00001)
    save_graph(fp_ratio_F, tp_ratio_F, filename, epoch)
    auc_F = sm.auc(fp_ratio_F, tp_ratio_F)
    
    argsort_S = np.argsort(score_A_np, axis=0)[:, 1]
    score_A_np_sort_S = score_A_np[argsort_S][::-1]
    value_1_0_S = (np.where(score_A_np_sort_S[:, 2] == threshold_tau, 1., 0.)).astype(np.float32)
    # score_A_np_sort_0_1 = np.concatenate((score_A_np_sort, value_1_0), axis=1)
    sum_1_S = np.sum(value_1_0_S)
    len_s_S = len(score_A_np)
    sum_0_S = len_s_S - sum_1_S
    tp_S = np.cumsum(value_1_0_S).astype(np.float32)
    index_S = np.arange(1, len_s_S + 1, 1).astype(np.float32)
    fp_S = index_S - tp_S
    fn_S = sum_1_S - tp_S
    tn_S = sum_0_S - fp_S
    tp_ratio_S = tp_S / (tp_S + fn_S + 0.00001)
    fp_ratio_S = fp_S / (fp_S + tn_S + 0.00001)
    save_graph(fp_ratio_S, tp_ratio_S, filename, epoch)
    auc_S = sm.auc(fp_ratio_S, tp_ratio_S)
    
    return auc_F, auc_S


def unnorm_img(img_np):
    img_np_255 = (img_np + 1.0) * 127.5
    img_np_255_mod1 = np.maximum(img_np_255, 0)
    img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
    img_np_uint8 = img_np_255_mod1.astype(np.uint8)
    return img_np_uint8


def convert_np2pil(images_255):
    list_images_PIL = []
    for num, images_255_1 in enumerate(images_255):
        # img_255_tile = np.tile(images_255_1, (1, 1, 3))
        image_1_PIL = Image.fromarray(images_255_1)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL
    
def make_output_img(img_batch_5, img_batch_7, x_z_x_5, x_z_x_7, epoch, log_file_name, out_img_dir):
    (data_num, img1_h, img1_w, _) = img_batch_5.shape
    img_batch_5_mono = np.mean(img_batch_5, axis=3).reshape(img_batch_5.shape[0], img_batch_5.shape[1], img_batch_5.shape[2], 1)
    img_batch_7_mono = np.mean(img_batch_7, axis=3).reshape(img_batch_7.shape[0], img_batch_7.shape[1], img_batch_7.shape[2], 1)
    x_z_x_5_mono = np.mean(x_z_x_5, axis=3).reshape(x_z_x_5.shape[0], x_z_x_5.shape[1], x_z_x_5.shape[2], 1)
    x_z_x_7_mono = np.mean(x_z_x_7, axis=3).reshape(x_z_x_7.shape[0], x_z_x_7.shape[1], x_z_x_7.shape[2], 1)

    img_batch_5_unn = np.tile(unnorm_img(img_batch_5_mono), (1, 1, 3))
    img_batch_7_unn = np.tile(unnorm_img(img_batch_7_mono), (1, 1, 3))
    x_z_x_5_unn = np.tile(unnorm_img(x_z_x_5_mono), (1, 1, 3))
    x_z_x_7_unn = np.tile(unnorm_img(x_z_x_7_mono), (1, 1, 3))

    diff_5 = img_batch_5_mono - x_z_x_5_mono
    diff_5_r = (2.0 * np.maximum(diff_5, 0.0)) - 1.0 #(0.0, 1.0) -> (-1.0, 1.0)
    diff_5_b = (2.0 * np.abs(np.minimum(diff_5, 0.0))) - 1.0 #(-1.0, 0.0) -> (1.0, 0.0) -> (1.0, -1.0)
    diff_5_g = diff_5_b * 0.0 - 1.0
    diff_5_r_unnorm = unnorm_img(diff_5_r)
    diff_5_b_unnorm = unnorm_img(diff_5_b)
    diff_5_g_unnorm = unnorm_img(diff_5_g)
    diff_5_np = np.concatenate((diff_5_r_unnorm, diff_5_g_unnorm, diff_5_b_unnorm), axis=3)
    
    diff_7 = img_batch_7_mono - x_z_x_7_mono
    diff_7_r = (2.0 * np.maximum(diff_7, 0.0)) - 1.0 #(0.0, 1.0) -> (-1.0, 1.0)
    diff_7_b = (2.0 * np.abs(np.minimum(diff_7, 0.0))) - 1.0 #(-1.0, 0.0) -> (1.0, 0.0) -> (1.0, -1.0)
    diff_7_g = diff_7_b * 0.0 - 1.0
    diff_7_r_unnorm = unnorm_img(diff_7_r)
    diff_7_b_unnorm = unnorm_img(diff_7_b)
    diff_7_g_unnorm = unnorm_img(diff_7_g)
    diff_7_np = np.concatenate((diff_7_r_unnorm, diff_7_g_unnorm, diff_7_b_unnorm), axis=3)

    img_batch_5_PIL = convert_np2pil(img_batch_5_unn)
    img_batch_7_PIL = convert_np2pil(img_batch_7_unn)
    x_z_x_5_PIL = convert_np2pil(x_z_x_5_unn)
    x_z_x_7_PIL = convert_np2pil(x_z_x_7_unn)
    diff_5_PIL = convert_np2pil(diff_5_np)
    diff_7_PIL = convert_np2pil(diff_7_np)

    wide_image_np = np.ones(((img1_h + 1) * data_num - 1, (img1_w + 1) * 6 - 1, 3), dtype=np.uint8) * 255
    wide_image_PIL = Image.fromarray(wide_image_np)
    for num, (ori_5, ori_7, xzx5, xzx7, diff5, diff7) in enumerate(zip(img_batch_5_PIL, img_batch_7_PIL ,x_z_x_5_PIL, x_z_x_7_PIL, diff_5_PIL, diff_7_PIL)):
        wide_image_PIL.paste(ori_5, (0, num * (img1_h + 1)))
        wide_image_PIL.paste(xzx5, (img1_w + 1, num * (img1_h + 1)))
        wide_image_PIL.paste(diff5, ((img1_w + 1) * 2, num * (img1_h + 1)))
        wide_image_PIL.paste(ori_7, ((img1_w + 1) * 3, num * (img1_h + 1)))
        wide_image_PIL.paste(xzx7, ((img1_w + 1) * 4, num * (img1_h + 1)))
        wide_image_PIL.paste(diff7, ((img1_w + 1) * 5, num * (img1_h + 1)))

    wide_image_PIL.save(out_img_dir + "/resultImage_"+ log_file_name + '_' + str(epoch) + ".png")

def save_list_to_csv(list, filename):
    f = open(filename, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(list)
    f.close()






