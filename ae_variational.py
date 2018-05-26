import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import ae_tools as tools
from ops import *


class VariationalAutoEncoder:

    # net
    def __init__(self, n_input, n_hidden = 20, batch_size = 256, optimizer=tf.train.AdamOptimizer(learning_rate=0.001)):
        self.n_input = n_input

        # 隐变量的个数是自己确定的
        self.n_hidden = n_hidden

        self.images = tf.placeholder(tf.float32, [None, 784])
        # 获得数据量
        self.batch_size = batch_size
        image_matrix = tf.reshape(self.images, [-1, 28, 28, 1])

        # encoder
        z_mean, z_stddev = self.encoder(image_matrix)
        # 高斯化
        samples = tf.random_normal([self.batch_size, self.n_hidden], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)
        # decoder
        self.generated_images = self.decoder(guessed_z)
        self.output = tf.reshape(self.generated_images, [self.batch_size, 28 * 28])

        # 重构loss： 网络重构图片的能力
        # 交叉熵损失
        self.reconstr_loss = - tf.reduce_sum(
            self.images * tf.log(1e-8 + self.output) + (1 - self.images) * tf.log(1e-8 + 1 - self.output), 1)
        # MSE损失
        # self.reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.output, self.images), 2.0))
        # a latent loss： KL散度度量下隐变量和单位高斯分布的接近程度
        self.latent_loss = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)
        self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss)

        self.nan_1 = tf.reduce_mean(tf.log(1e-8 + self.output))
        self.nan_2 = tf.reduce_mean(tf.log(1e-8 + 1 - self.output))
        self.nan_3 = tf.reduce_mean(tf.log(tf.square(z_stddev)))
        self.loss_1 = tf.reduce_mean(self.reconstr_loss)
        self.loss_2 = tf.reduce_mean(self.latent_loss)

        # 优化
        self.optimizer = optimizer.minimize(self.cost)

        # sess
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())

    # train
    def partial_fit(self, X):
        cost, _, loss_1, loss_2 = self.sess.run([self.cost, self.optimizer, self.reconstr_loss, self.latent_loss],
                                                feed_dict={self.images: X})
        return cost

    def partial_fit_for_nan(self, X):
        cost, _, loss_1, loss_2, nan_1, nan_2, nan_3 = self.sess.run([self.cost, self.optimizer, self.loss_1, self.loss_2,
                                                 self.nan_1, self.nan_2, self.nan_3], feed_dict={self.images: X})
        return cost, loss_1, loss_2, nan_1, nan_2, nan_3

    def calculate_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.images: X})

    # hidden -> output
    def generate(self, hidden=None, batch_size=10):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([batch_size, self.n_hidden]))
        return self.sess.run(self.output, feed_dict={self.hiden: hidden})

    # input -> output
    def output_result(self, X):
        return self.sess.run(self.output, feed_dict={self.images: X})

    # encoder
    def encoder(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1"))  # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2"))  # 14x14x16 -> 7x7x32
            h2 = lrelu(conv2d(h2, 32, 64, "d_h3"))  # 7x7x32 -> 4x4x64
            h2 = lrelu(conv2d(h2, 64, 128, "d_h4"))  # 4x4x64 -> 2x2x128
            h2_flat = tf.reshape(h2, [self.batch_size, 2 * 2 * 128])
            # h2_flat = tf.reshape(h2, [self.batch_size, 7 * 7 * 32])

            w_mean = dense(h2_flat, 2 * 2 * 128, self.n_hidden, "w_mean")
            w_stddev = dense(h2_flat, 2 * 2 * 128, self.n_hidden, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def decoder(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_hidden, 4 * 4 * 64, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batch_size, 4, 4, 64]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batch_size, 7, 7, 32], "g_h1"))
            h1 = tf.nn.relu(conv_transpose(h1, [self.batch_size, 14, 14, 16], "g_h2"))
            h2 = conv_transpose(h1, [self.batch_size, 28, 28, 1], "g_h3")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def decoder_4(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_hidden, 2 * 2 * 128, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batch_size, 2, 2, 128]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batch_size, 4, 4, 64], "g_h4"))
            h1 = tf.nn.relu(conv_transpose(h1, [self.batch_size, 7, 7, 32], "g_h7"))
            h1 = tf.nn.relu(conv_transpose(h1, [self.batch_size, 14, 14, 16], "g_h14"))
            h2 = conv_transpose(h1, [self.batch_size, 28, 28, 1], "g_h0")
            h2 = tf.nn.sigmoid(h2)

        return h2

    pass


class Runner:

    def __init__(self, autoencoder):
        self.autoencoder = autoencoder
        self.mnist = input_data.read_data_sets("data/MNIST_data", one_hot=True)
        self.x_train, self.x_test = tools.minmax_scale(self.mnist.train.images, self.mnist.test.images)
        self.train_number = self.mnist.train.num_examples

    def train(self, train_epochs=2000, display_step=1, save_step = 10):

        batch_size = self.autoencoder.batch_size

        for epoch in range(train_epochs):

            avg_cost = 0.
            avg_KL = 0.
            avg_reconstr = 0.
            total_batch = int(self.train_number) // batch_size

            for i in range(total_batch):
                batch_xs = tools.get_random_block_from_data(self.x_train, batch_size)

                cost, reconstr_cost, Kl_cost, nan_1, nan_2, nan_3 = self.autoencoder.partial_fit_for_nan(batch_xs)

                avg_cost += cost / self.train_number * batch_size
                avg_KL += Kl_cost / self.train_number * batch_size
                avg_reconstr += reconstr_cost / self.train_number * batch_size

                # print(time.strftime("%H:%M:%S", time.localtime()),
                #       "Epoch:{} cost={:.9f} KL_cost = {:.9f}, reconstr_cost = {:.9} nan_1 = {:.9} nan_2 = {:.9}, nan_3 = {:.9}"
                #       .format(epoch + 1, avg_cost, Kl_cost,reconstr_cost, nan_1, nan_2, nan_3))
            # diaplay
            if epoch % display_step == 0:
                print(time.strftime("%H:%M:%S", time.localtime()),
                      "Epoch:{} cost={:.9f} KL_cost = {:.9f}, reconstr_cost = {:.9}".format(epoch + 1, avg_cost, avg_KL, avg_reconstr))
            # save
            if epoch % save_step == 0:
                self.save_result(
                    file_name="result-{}-{}-{}-{}".format(epoch, self.autoencoder.n_input,
                                                          self.autoencoder.n_hidden, avg_cost))

        print(time.strftime("%H:%M:%S", time.localtime()), "Total cost: {}".format(self.autoencoder.calculate_total_cost(self.mnist.test.images)))

        pass

    def save_result(self, file_name, n_show=10):
        # 显示编码结果和解码后结果
        images = tools.get_random_block_from_data(self.x_test, self.autoencoder.batch_size, fixed=True)
        decode = self.autoencoder.output_result(images)
        # 对比原始图片重建图片
        tools.gaussian_save_result(images[:n_show], decode[:n_show], decode[:n_show],
                                   save_path="result/ae-variational/{}.jpg".format(file_name))
        pass

    pass

if __name__ == '__main__':
    runner = Runner(autoencoder=VariationalAutoEncoder(batch_size=256, n_input=784, n_hidden=20))
    runner.train()

    pass
