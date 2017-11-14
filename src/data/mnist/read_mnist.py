import os
import struct
from array import array
import random
import numpy as np

class MNIST():
    def __init__(self, path):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

        self.num_classes = 10

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = self.process_images(ims)
        self.test_labels, self.one_hot_test_labels = self.process_labels(labels)

        return self.test_images, self.test_labels, self.one_hot_test_labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = self.process_images(ims)
        self.train_labels, self.one_hot_train_labels = self.process_labels(labels)

        return self.train_images, self.train_labels, self.one_hot_train_labels

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        one_hot_labels = np.eye(self.num_classes, dtype=float)[labels]
        return np.array(labels), one_hot_labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    def transfer_learning_mnist(self, train_limit, test_limit):
        self.load_testing()
        self.load_training()

        self.x_train_lt5 = self.train_images[self.train_labels < 5]
        self.y_train_lt5 = self.train_labels[self.train_labels < 5]
        shuffle = np.random.permutation(len(self.y_train_lt5))
        self.x_train_lt5, self.y_train_lt5 = self.x_train_lt5[shuffle][:train_limit], self.y_train_lt5[shuffle][:train_limit]

        self.x_test_lt5 = self.test_images[self.test_labels < 5]
        self.y_test_lt5 = self.test_labels[self.test_labels < 5]
        shuffle = np.random.permutation(len(self.y_test_lt5))
        self.x_test_lt5, self.y_test_lt5 = self.x_test_lt5[shuffle][:test_limit], self.y_test_lt5[shuffle][:test_limit]

        self.x_train_gte5 = self.train_images[self.train_labels >= 5]
        self.y_train_gte5 = self.train_labels[self.train_labels >= 5] - 5
        shuffle = np.random.permutation(len(self.y_train_gte5))
        self.x_train_gte5, self.y_train_gte5 = self.x_train_gte5[shuffle][:train_limit], self.y_train_gte5[shuffle][:train_limit]

        self.x_test_gte5 = self.test_images[self.test_labels >= 5]
        self.y_test_gte5 = self.test_labels[self.test_labels >= 5] - 5
        shuffle = np.random.permutation(len(self.y_test_gte5))
        self.x_test_gte5, self.y_test_gte5 = self.x_test_gte5[shuffle][:test_limit], self.y_test_gte5[shuffle][:test_limit]

        return (self.x_train_lt5, self.y_train_lt5), (self.x_test_lt5, self.y_test_lt5), (self.x_train_gte5, self.y_train_gte5), (self.x_test_gte5, self.y_test_gte5)
