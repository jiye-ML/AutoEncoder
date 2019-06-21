import tensorflow as tf
import time
from sklearn import preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 数据
class MnistData:
    def __init__(self):
        # 手写体数据
        self._mnist = input_data.read_data_sets("data/MNIST_data", one_hot=False)
        # 初始数据
        self.x_train, self.x_test = self.standard_scale(self._mnist.train.images, self._mnist.test.images)
        self.y_train, self.y_test = self._mnist.train.labels, self._mnist.test.labels
        # 网络中的数据
        self.x_train_input = self.x_train
        # 数据的数量
        self.train_number = self._mnist.train.num_examples
        self.test_number = self._mnist.test.num_examples
        pass

    # 数据标准化
    @staticmethod
    def standard_scale(x_train, x_test):
        preprocessor = prep.StandardScaler().fit(x_train)
        return preprocessor.transform(x_train), preprocessor.transform(x_test)

    # 训练时获取下一批数据
    def nest_batch_train(self, batch_size):
        start_id = np.random.randint(0, len(self.x_train) - batch_size)
        return self.x_train_input[start_id: start_id + batch_size], self.y_train[start_id: start_id + batch_size]

    # 微调时获取下一批数据
    def nest_batch_fine(self, batch_size):
        start_index = np.random.randint(0, len(self.x_train) - batch_size)
        return self.x_train[start_index: start_index + batch_size], self.y_train[start_index: start_index + batch_size]

    # 获取下一批测试数据
    def nest_batch_test(self, batch_size, fixed=False):
        start_index = 0 if fixed else np.random.randint(0, len(self.x_test) - batch_size)
        return self.x_test[start_index: start_index + batch_size], self.y_test[start_index: start_index + batch_size]

    pass


# 堆栈自编码网络
class StackAutoEncoder:
    # 定义网络
    def __init__(self, layer_1_n_hidden, layer_2_n_hidden, type_number=10, data_n_input=784):
        self.type_number = type_number
        self.data_n_input = data_n_input
        self.layer_1_n_hidden = layer_1_n_hidden
        self.layer_2_n_hidden = layer_2_n_hidden

        self._activation = tf.nn.softplus
        self._optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self._weights = self._initialize_weights()

        self.layer_1_x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_n_input])
        self.layer_2_x = tf.placeholder(dtype=tf.float32, shape=[None, self.layer_1_n_hidden])
        self.classifies_x = tf.placeholder(dtype=tf.float32, shape=[None, self.layer_2_n_hidden])
        self.classifies_y = tf.placeholder(dtype=tf.int32, shape=[None])

        # 第一层
        self.layer_1_hidden, _, self.layer_1_cost, self.layer_1_optimizer = self._auto_encoder(
            self.layer_1_x, self._weights["layer_1_w1"], self._weights["layer_1_b1"],
            self._weights["layer_1_w2"], self._weights["layer_1_b2"])
        # 第二层
        self.layer_2_hidden, _, self.layer_2_cost, self.layer_2_optimizer = self._auto_encoder(
            self.layer_2_x, self._weights["layer_2_w1"], self._weights["layer_2_b1"],
            self._weights["layer_2_w2"], self._weights["layer_2_b2"])
        # 分类器
        self.classifies_cost, self.classifies_optimizer, self.classifies_result = \
            self._classifies(self.classifies_x, self.classifies_y,
                             self._weights["classifies_1_w1"], self._weights["classifies_1_b1"])

        # 微调
        self.last_cost, self.last_optimizer, self.last_result = \
            self._fine_tuning(self.layer_1_x, self.classifies_y, self._weights["layer_1_w1"],
                              self._weights["layer_1_b1"], self._weights["layer_2_w1"], self._weights["layer_2_b1"],
                              self._weights["classifies_1_w1"], self._weights["classifies_1_b1"])

        # session
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())

        # 计算参数个数
        self.para_number = self._para_number()
        pass

    # 所有权值
    def _initialize_weights(self):
        all_weights = dict()
        # 第一层
        all_weights["layer_1_w1"] = tf.get_variable("layer_1_w1", shape=[self.data_n_input, self.layer_1_n_hidden],
                                                    initializer=tf.contrib.layers.xavier_initializer())
        all_weights["layer_1_b1"] = tf.Variable(tf.zeros([self.layer_1_n_hidden], dtype=tf.float32))
        all_weights["layer_1_w2"] = tf.Variable(tf.zeros([self.layer_1_n_hidden, self.data_n_input], dtype=tf.float32))
        all_weights["layer_1_b2"] = tf.Variable(tf.zeros([self.data_n_input]), dtype=tf.float32)

        # 第二层
        all_weights["layer_2_w1"] = tf.get_variable("layer_2_w1", shape=[self.layer_1_n_hidden, self.layer_2_n_hidden],
                                                    initializer=tf.contrib.layers.xavier_initializer())
        all_weights["layer_2_b1"] = tf.Variable(tf.zeros([self.layer_2_n_hidden], dtype=tf.float32))
        all_weights["layer_2_w2"] = tf.Variable(tf.zeros([self.layer_2_n_hidden, self.layer_1_n_hidden],
                                                         dtype=tf.float32))
        all_weights["layer_2_b2"] = tf.Variable(tf.zeros([self.layer_1_n_hidden]), dtype=tf.float32)

        # 分类器
        all_weights["classifies_1_w1"] = tf.Variable(tf.zeros([self.layer_2_n_hidden, self.type_number]),
                                                     dtype=tf.float32)
        all_weights["classifies_1_b1"] = tf.Variable(tf.zeros([self.type_number]), dtype=tf.float32)

        return all_weights


    # 计算参数个数
    def _para_number(self):
        para_number = 0
        for key, value in self._weights.items():
            shape = self.sess.run(tf.shape(value))
            now_count = shape[0]
            for i in range(1, len(shape)):
                now_count *= shape[i]
            para_number += now_count
        return para_number

    # 自编码
    def _auto_encoder(self, x, w1, b1, w2=None, b2=None, is_fine=False):
        # model
        hidden = self._activation(tf.add(tf.matmul(x, w1), b1))
        if is_fine:
            return hidden
        else:
            if w2 is None or b2 is None:
                raise Exception("auto_encoder() 缺少参数")

            output = tf.add(tf.matmul(hidden, w2), b2)
            # cost
            cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(output, x), 2.0))
            optimizer = self._optimizer.minimize(cost)
            return hidden, output, cost, optimizer

    # 分类器
    def _classifies(self, x, y, w1, b1):
        # model
        output = tf.add(tf.matmul(x, w1), b1)
        # cost
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y, tf.int64), logits=output)
        cost = tf.reduce_mean(cross_entropy)
        # optimizer
        optimizer = self._optimizer.minimize(cost)
        # 结果
        result = tf.argmax(output, 1)
        return cost, optimizer, result

    # 微调
    def _fine_tuning(self, x, y, layer_1_w1, layer_1_b1, layer_2_w1, layer_2_b1, classifies_1_w1, classifies_1_b1):
        layer_1_hidden = self._auto_encoder(x, layer_1_w1, layer_1_b1, is_fine=True)
        layer_2_hidden = self._auto_encoder(layer_1_hidden, layer_2_w1, layer_2_b1, is_fine=True)
        cost, optimizer, result = self._classifies(layer_2_hidden, y, classifies_1_w1, classifies_1_b1)
        return cost, optimizer, result
        pass

    pass


# 运行器
class Runner:

    def __init__(self, data, auto_encoder):
        self.data = data
        self.auto_encoder = auto_encoder
        pass

    # 训练
    def train(self, train_layer_1_epochs, train_layer_2_epochs, train_classifies_epochs, batch_size=64):
        total_batch = int(self.data.train_number) // batch_size

        # 参数个数
        self.print_info("the parameter number is {}".format(self.auto_encoder.para_number))

        # 第一层训练
        self.print_info("begin to train first layers")
        for epoch in range(train_layer_1_epochs):
            avg_cost = 0.
            for i in range(total_batch):
                batch_xs, _ = self.data.nest_batch_train(batch_size)
                cost, _ = self.auto_encoder.sess.run(
                    [self.auto_encoder.layer_1_cost, self.auto_encoder.layer_1_optimizer],
                    feed_dict={self.auto_encoder.layer_1_x: batch_xs})
                avg_cost += cost / self.data.train_number * batch_size
            self.print_info("{} cost: {}".format(epoch + 1, avg_cost))
            pass
        self.print_info("first layers train end")

        # 得到第一层的输出
        self.print_info("begin to get the output of first layers")
        self.data.x_train_input = self.auto_encoder.sess.run(self.auto_encoder.layer_1_hidden, feed_dict={
            self.auto_encoder.layer_1_x: self.data.x_train_input})
        self.print_info("get the output of first layers end")

        # 第二层训练
        self.print_info("begin to train second layers")
        for epoch in range(train_layer_2_epochs):
            avg_cost = 0.
            for i in range(total_batch):
                batch_xs, _ = self.data.nest_batch_train(batch_size)
                cost, _ = self.auto_encoder.sess.run(
                    [self.auto_encoder.layer_2_cost, self.auto_encoder.layer_2_optimizer],
                    feed_dict={self.auto_encoder.layer_2_x: batch_xs})
                avg_cost += cost / self.data.train_number * batch_size
            self.print_info("{} cost: {}".format(epoch + 1, avg_cost))
            pass
        self.print_info("second layers train end")

        # 得到第二层的输出
        self.print_info("begin to get the output of second layers")
        self.data.x_train_input = self.auto_encoder.sess.run(self.auto_encoder.layer_2_hidden, feed_dict={
            self.auto_encoder.layer_2_x: self.data.x_train_input})
        self.print_info("get the output of second layers end")

        # 分类层
        self.print_info("begin to train classifies layers")
        for epoch in range(train_classifies_epochs):
            accuracy = 0.
            for i in range(total_batch):
                batch_xs, batch_ys = self.data.nest_batch_train(batch_size)
                _, _, result = self.auto_encoder.sess.run(
                    [self.auto_encoder.classifies_cost, self.auto_encoder.classifies_optimizer,
                     self.auto_encoder.classifies_result],
                    feed_dict={self.auto_encoder.classifies_x: batch_xs, self.auto_encoder.classifies_y: batch_ys})
                accuracy += self.stat_accuracy(result, batch_ys)
                pass
            self.print_info(
                "after {} accuracy: {} ({}/{})".format(epoch + 1, accuracy / (total_batch * batch_size), accuracy,
                                                       total_batch * batch_size))
            pass
        self.print_info("classifies layers train end")

        # 对分类进行测试
        self.print_info("begin to test classifies layers")
        result = self.auto_encoder.sess.run(self.auto_encoder.classifies_result,
                                            feed_dict={self.auto_encoder.classifies_x: self.data.x_train_input})
        accuracy_number = self.stat_accuracy(result, self.data.y_train)
        self.print_info("now accuracy is {} ({}/{})".format(accuracy_number / self.data.train_number, accuracy_number,
                                                            self.data.train_number))
        self.print_info("test classifies layers end")

        self.print_info("all train end")
        pass

    # 测试
    def fine_tuning(self, fine_epochs, batch_size=64):
        total_batch = int(self.data.train_number) // batch_size

        # 微调
        self.print_info("begin to fine tuning")
        for epoch in range(fine_epochs):
            avg_cost = 0.
            accuracy = 0.
            for i in range(total_batch):
                batch_xs, batch_ys = self.data.nest_batch_fine(batch_size)
                cost, _, result = self.auto_encoder.sess.run(
                    [self.auto_encoder.last_cost, self.auto_encoder.last_optimizer, self.auto_encoder.last_result],
                    feed_dict={self.auto_encoder.layer_1_x: batch_xs, self.auto_encoder.classifies_y: batch_ys})
                avg_cost += cost / self.data.train_number * batch_size
                accuracy += self.stat_accuracy(result, batch_ys)
            self.print_info("{} cost: {}".format(epoch + 1, avg_cost))
            self.print_info("now accuracy is {} ({}/{})".format(accuracy / (total_batch * batch_size), accuracy,
                                                                total_batch * batch_size))
            pass
        self.print_info("fine tuning end")

        # 测试
        self.print_info("begin to test")
        result = self.auto_encoder.sess.run(self.auto_encoder.last_result,
                                            feed_dict={self.auto_encoder.layer_1_x: self.data.x_train})
        accuracy_number = self.stat_accuracy(result, self.data.y_train)
        self.print_info("now accuracy is {} ({}/{})".format(accuracy_number / self.data.train_number, accuracy_number,
                                                            self.data.train_number))
        self.print_info("test end")
        pass

    # 测试
    def test(self):
        self.print_info("begin to test")
        result = self.auto_encoder.sess.run(self.auto_encoder.last_result,
                                            feed_dict={self.auto_encoder.layer_1_x: self.data.x_test})
        accuracy_number = self.stat_accuracy(result, self.data.y_test)
        self.print_info("now accuracy is {} ({}/{})".format(accuracy_number / self.data.test_number, accuracy_number,
                                                            self.data.test_number))
        self.print_info("test end")
        pass

    # 统计正确的个数
    @staticmethod
    def stat_accuracy(labels_1, labels_2):
        return np.sum(np.equal(labels_1, labels_2))

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)

    pass


def train_test_fine_test(layer_1_n=200, layer_2_n=50, layer_1_epochs=20, layer_2_epochs=10, classifies_epochs=10,
                         fine_epochs=100):
    runner = Runner(auto_encoder=StackAutoEncoder(layer_1_n_hidden=layer_1_n,
                                                  layer_2_n_hidden=layer_2_n), data=MnistData())
    print()
    runner.train(train_layer_1_epochs=layer_1_epochs, train_layer_2_epochs=layer_2_epochs,
                 train_classifies_epochs=classifies_epochs)
    print()
    runner.test()
    print()
    runner.fine_tuning(fine_epochs=fine_epochs)
    print()
    runner.test()
    pass


def test_fine_test(layer_1_n=200, layer_2_n=50, fine_epochs=100):
    runner = Runner(auto_encoder=StackAutoEncoder(layer_1_n_hidden=layer_1_n,
                                                  layer_2_n_hidden=layer_2_n), data=MnistData())
    print()
    runner.test()
    print()
    runner.fine_tuning(fine_epochs=fine_epochs)
    print()
    runner.test()
    pass


if __name__ == '__main__':
    # 没有经过预训练
    # test_fine_test()

    print("------------------------------------------------------------------")

    # 先预训练、微调，再测试
    train_test_fine_test(layer_1_epochs = 300, layer_2_epochs=300, classifies_epochs=300)
    pass
