import numpy as np
import os
import random

class Make_datasets_MNIST():

    def __init__(self, file_name, img_width, img_height, seed, inlier_num=1):
        self.filename = file_name
        self.img_width = img_width
        self.img_height = img_height
        self.inlier_num = inlier_num
        self.seed = seed
        x_train, x_test, x_valid, y_train, y_test, y_valid = self.read_MNIST_npy(self.filename)
        self.train_np = np.concatenate((y_train.reshape(-1,1), x_train), axis=1).astype(np.float32)
        self.test_np = np.concatenate((y_test.reshape(-1,1), x_test), axis=1).astype(np.float32)
        self.valid_np = np.concatenate((y_valid.reshape(-1,1), x_valid), axis=1).astype(np.float32)
        print("self.train_np.shape, ", self.train_np.shape)
        print("self.test_np.shape, ", self.test_np.shape)
        print("self.valid_np.shape, ", self.valid_np.shape)
        print("np.max(x_train), ", np.max(x_train))
        print("np.min(x_train), ", np.min(x_train))
        self.train_inlier, self.train_outlier = self.divide_MNIST_by_digit(self.train_np, self.inlier_num)
        print("self.train_data_5.shape, ", self.train_inlier.shape)
        print("self.train_data_7.shape, ", self.train_outlier.shape)
        self.valid_inlier, self.valid_outlier = self.divide_MNIST_by_digit(self.valid_np, self.inlier_num)
        print("self.valid_inlier.shape, ", self.valid_inlier.shape)
        print("self.valid_outlier.shape, ", self.valid_outlier.shape)
        self.valid_all = np.concatenate((self.train_outlier, self.valid_outlier, self.valid_inlier))
        print("self.valid_all.shape, ", self.valid_all.shape)
        random.seed(self.seed)
        np.random.seed(self.seed)


    def read_MNIST_npy(self, filename):
        mnist_npz = np.load(filename)
        print("type(mnist_npz), ", type(mnist_npz))
        print("mnist_npz.keys(), ", mnist_npz.keys())
        print("mnist_npz['x_train'].shape, ", mnist_npz['x_train'].shape)
        print("mnist_npz['x_test'].shape, ", mnist_npz['x_test'].shape)
        print("mnist_npz['x_valid'].shape, ", mnist_npz['x_valid'].shape)
        print("mnist_npz['y_train'].shape, ", mnist_npz['y_train'].shape)
        print("mnist_npz['y_test'].shape, ", mnist_npz['y_test'].shape)
        print("mnist_npz['y_valid'].shape, ", mnist_npz['y_valid'].shape)
        x_train = mnist_npz['x_train']
        x_test = mnist_npz['x_test']
        x_valid = mnist_npz['x_valid']
        y_train = mnist_npz['y_train']
        y_test = mnist_npz['y_test']
        y_valid = mnist_npz['y_valid']
        return x_train, x_test, x_valid, y_train, y_test, y_valid


    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files

    def divide_MNIST_by_digit(self, train_np, inlier_num):
        data_inlier = train_np[train_np[:,0] == inlier_num]
        data_outlier = train_np[train_np[:,0] != inlier_num]
        return data_inlier, data_outlier


    def read_data(self, d_y_np, width, height):
        tars = []
        images = []
        for num, d_y_1 in enumerate(d_y_np):
            image = d_y_1[1:].reshape(width, height, 1)
            image = np.tile(image, (1,1,3))
            tar = d_y_1[0]
            images.append(image)
            tars.append(tar)

        return np.asarray(images), np.asarray(tars)


    def normalize_data(self, data):
        # data0_2 = data / 127.5
        # data_norm = data0_2 - 1.0
        data_norm = (data * 2.0) - 1.0 #applied for tanh

        return data_norm


    def make_data_for_1_epoch(self):
        self.filename_1_epoch = np.random.permutation(self.train_inlier)

        return len(self.filename_1_epoch)


    def get_data_for_1_batch(self, i, batchsize):
        filename_batch = self.filename_1_epoch[i:i + batchsize]
        images, _ = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n


    def get_valid_data_for_1_batch(self, i, batchsize):
        filename_batch = self.valid_all[i:i + batchsize]
        images, tars = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n, tars


    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        norms = np.random.normal(mean, stddev, (data_num, unit_num))
        # tars = np.zeros((data_num, 1), dtype=np.float32)
        return norms


    def make_target_1_0(self, value, data_num):
        if value == 0.0:
            target = np.zeros((data_num, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, 1), dtype=np.float32)
        else:
            print("target value error")
            target = None
        return target

def check_mnist_npz(filename):
    mnist_npz = np.load(filename)
    print("type(mnist_npz), ", type(mnist_npz))
    print("mnist_npz.keys(), ", mnist_npz.keys())
    print("mnist_npz['x_train'].shape, ", mnist_npz['x_train'].shape)
    print("mnist_npz['x_test'].shape, ", mnist_npz['x_test'].shape)
    print("mnist_npz['x_valid'].shape, ", mnist_npz['x_valid'].shape)
    print("mnist_npz['y_train'].shape, ", mnist_npz['y_train'].shape)
    print("mnist_npz['y_test'].shape, ", mnist_npz['y_test'].shape)
    print("mnist_npz['y_valid'].shape, ", mnist_npz['y_valid'].shape)


if __name__ == '__main__':
    #debug
    FILE_NAME = './mnist.npz'
    # check_mnist_npz(FILE_NAME)
    # make_datasets = Make_datasets_MNIST(FILE_NAME, 28, 28, 1234)
