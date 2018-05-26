'''
去燥自编码网络
单层神经网络架构
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import ae_tools as tool


class AdditiveGaussianNoiseAutoencoder(object):

    def __init__(self, n_input, n_hidden, activation_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(),
                 scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activation = activation_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # net
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.activation(
            tf.add(tf.matmul(self.x + scale * tf.random_normal([n_input]), self.weights['w1']), self.weights['b1'])
        )
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # loss
        self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))

        # optimize
        self.optimizer = optimizer.minimize(self.cost)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())

        pass

    # 初始化权值
    def _initialize_weights(self):
        all_weights = dict()
        all_weights["w1"] = tf.Variable(tool.xavier_init(self.n_input, self.n_hidden))
        all_weights["b1"] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights["w2"] = tf.Variable(tool.xavier_init(self.n_hidden, self.n_input))
        all_weights["b2"] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights
        pass

    # 只执行一步的train
    def partial_fit(self, x):
        cost, opt = self.sess.run([self.cost, self.optimizer], feed_dict = {self.x: x, self.scale: self.training_scale})
        return cost
        pass

    # 计算cost
    def calculate_total_cost(self, x):
        return self.sess.run([self.cost], feed_dict= {self.x: x, self.scale: self.training_scale})

    # 获取学习到的高阶特征
    def transform(self, x):
        return self.sess.run(self.hidden, feed_dict= {self.x: x, self.scale: self.training_scale})
        pass

    # 将隐含层的输出结果作为输入， 通过之后的重建层将提取到的高阶特征复原为原始数据。
    def generate(self, hidden = None, batch_size = 10):
        if hidden is None:
            hidden = np.random.normal(size = [batch_size, self.n_hidden])
        # 人工喂入数据，阶段流程。
        return self.sess.run(self.reconstruction, feed_dict= {self.hidden: hidden})
        pass

    # 整体运行一遍复原过程，
    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict= {self.x: x, self.scale: self.training_scale})

    # 获得隐层 w1
    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    # 获得隐层 b1
    def get_biases(self):
        return self.sess.run(self.weights['b1'])

    pass


class Runner:

    def __init__(self, autoencoder):
        self.autoencoder = autoencoder
        # data
        self.mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

        self.x_train, self.x_test = tool.standard_scale(self.mnist.train.images, self.mnist.test.images)

        # 网络参数
        self.n_samples = int(self.mnist.train.num_examples)
        pass

    def fit(self, training_epoches = 20, batch_size = 128, display_step = 1):

        # train
        for epoch in range(training_epoches):
            avg_cost = 0.
            total_batch = self.n_samples // batch_size
            for i in range(total_batch):
                batch_xs = tool.get_random_block_from_data(self.x_train, batch_size)

                cost = self.autoencoder.partial_fit(batch_xs)
                avg_cost += cost / self.n_samples * batch_size

            if epoch % display_step == 0:
                self.save_result(file_name="{}-result-{}-{}-{}-{}".format(1, epoch, self.autoencoder.n_input, self.autoencoder.n_hidden, avg_cost))
                print("{} Epoch:{} cost={:.9f}".format(time.strftime("%H:%M:%S", time.localtime()), epoch + 1, avg_cost))

        pass

    def predict(self):
        print(time.strftime("%H:%M:%S", time.localtime()),
              "Total cost: {}".format(self.autoencoder.calculate_total_cost(self.x_test)))
        pass

    # 显示编码和解码后的结果
    def save_result(self, file_name, n_show = 15):
        images = tool.get_random_block_from_data(self.x_train, n_show, fixed=True)
        encode = self.autoencoder.transform(images)
        decode = self.autoencoder.generate(encode)

        # 对比原始图片重建图片
        tool.save_result(images, encode, decode, save_path="result/ae/{}/{}.jpg".format(self.autoencoder.n_hidden,
                                                                                        file_name))
        pass

    pass



if __name__ == '__main__':

    n_hiddens = [16, 32, 64, 80, 128, 256]

    for n_hidden in n_hiddens:
        # net
        autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, n_hidden = n_hidden,
                                                       activation_function = tf.nn.softplus,
                                                       optimizer = tf.train.AdamOptimizer(learning_rate= 0.001),
                                                       scale= 0.01)
        runner = Runner(autoencoder)
        runner.fit()

        runner.predict()

    pass
