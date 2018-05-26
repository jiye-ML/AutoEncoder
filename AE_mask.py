import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import ae_tools as tools


class MaskingAutoEncoder:

    # net
    def __init__(self, n_input, n_hidden, keep_prob=0.1, optimizer=tf.train.AdamOptimizer(learning_rate=0.001)):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activation = tf.nn.softplus
        self.keep_prob = keep_prob
        self.weights = self._initialize_weights()

        # model
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input])
        self.dropout = tf.nn.dropout(self.x, self.keep_prob)  # masking noise
        self.hidden = self.activation(tf.add(tf.matmul(self.dropout, self.weights["w1"]), self.weights["b1"]))
        self.output = tf.add(tf.matmul(self.hidden, self.weights["w2"]), self.weights["b2"])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.output, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        # sess
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())

    # all weights
    def _initialize_weights(self):
        all_weights = dict()
        all_weights["w1"] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        all_weights["b1"] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))

        all_weights["w2"] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights["b2"] = tf.Variable(tf.zeros([self.n_input]), dtype=tf.float32)
        return all_weights

    # train
    def partial_fit(self, X):
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.x: X})
        return cost

    def calculate_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    # hidden -> output
    def generate(self, hidden=None, batch_size=10):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([batch_size, self.n_hidden]))
        return self.sess.run(self.output, feed_dict={self.hidden: hidden})

    # input -> output
    def output_result(self, X):
        return self.sess.run([self.dropout, self.output], feed_dict={self.x: X})

    pass


class Runner:

    def __init__(self, autoencoder):
        self.autoencoder = autoencoder
        self.mnist = input_data.read_data_sets("data/MNIST_data", one_hot=True)
        self.x_train, self.x_test = tools.standard_scale(self.mnist.train.images, self.mnist.test.images)
        self.train_number = self.mnist.train.num_examples

    def train(self, train_epochs=2000, batch_size=256, display_step=1, save_step = 10):

        for epoch in range(train_epochs):

            avg_cost = 0.
            total_batch = int(self.train_number) // batch_size

            for i in range(total_batch):
                batch_xs = tools.get_random_block_from_data(self.x_test, batch_size)
                cost = self.autoencoder.partial_fit(batch_xs)
                avg_cost += cost / self.train_number * batch_size
            # print
            if epoch % display_step == 0:
                print(time.strftime("%H:%M:%S", time.localtime()), "Epoch:{}".format(epoch + 1), "cost={:.9f}".format(avg_cost))
            # save
            if epoch % save_step == 0:
                self.save_result(
                    file_name="result-{}-{}-{}-{}".format(epoch, self.autoencoder.n_input, self.autoencoder.n_hidden,
                                                      avg_cost))

        print(time.strftime("%H:%M:%S", time.localtime()), "Total cost: {}".format(self.autoencoder.calculate_total_cost(self.mnist.test.images)))

        pass

    def save_result(self, file_name, n_show=10):
        # 显示编码结果和解码后结果
        images = tools.get_random_block_from_data(self.x_test, n_show, fixed=True)
        masking_images, decode = self.autoencoder.output_result(images)
        # 对比原始图片重建图片
        tools.gaussian_save_result(images, masking_images, decode, save_path="result/ae-masking/{}.jpg".format(file_name))
        pass

    pass



if __name__ == '__main__':

    runner = Runner(autoencoder=MaskingAutoEncoder(n_input=784, n_hidden=256))
    runner.train(train_epochs = 500000, display_step = 10, save_step = 100)

    pass
