import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import ave_tools as tools


mnist = input_data.read_data_sets("data/", one_hot=True)

# 参数
learning_rate = 0.0001    # 学习速率
training_epochs = 500000  # 训练批次
batch_size = 256        # 随机选择训练数据大小
display_step = 50        # 展示步骤
save_step = 1000
examples_to_show = 10   # 显示示例图片数量


# 网络参数
n_hidden_1 = 256  # 第一隐层神经元数量
n_hidden_2 = 128  # 第二
n_hidden_3 = 64  # 第三
n_input = 784     # 输入

# 权重初始化
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input,    n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

# 偏置值初始化
biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.zeros([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.zeros([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.zeros([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.zeros([n_input])),
}

# 编码
def encoder(x):
    encode1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    encode2 = tf.nn.relu(tf.add(tf.matmul(encode1, weights['encoder_h2']), biases['encoder_b2']))
    encode3 = tf.nn.relu(tf.add(tf.matmul(encode2, weights['encoder_h3']), biases['encoder_b3']))
    return encode3

# 解码
def decoder(encoder):
    decoder1 = tf.nn.relu(tf.add(tf.matmul(encoder, weights['decoder_h1']), biases['decoder_b1']))
    decoder2 = tf.nn.relu(tf.add(tf.matmul(decoder1, weights['decoder_h2']), biases['decoder_b2']))
    decoder3 = tf.nn.relu(tf.add(tf.matmul(decoder2, weights['decoder_h3']), biases['decoder_b3']))
    #
    return tf.div(decoder3, tf.reduce_max(decoder3))


x = tf.placeholder(tf.float32, [None, n_input])

# 构造模型
encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

# 预测
y_pred = decoder_op
# 实际输入数据当作标签
y_true = x

# 定义代价函数和优化器，最小化平方误差,这里可以根据实际修改误差模型
cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(cost)

# 运行Graph
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 总的batch
    total_batch = int(mnist.train.num_examples/batch_size)
    # 开始训练
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})

        # 展示每次训练结果
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

        if epoch % save_step == 0:
            encodes, encode_decode, total_cost = sess.run([encoder_op, y_pred, cost], feed_dict={x: mnist.validation.images})
            print(time.strftime("%H:%M:%S", time.localtime()), "Total cost: {}".format(total_cost))
            # 对比原始图片重建图片
            tools.save_result(mnist.validation.images[:examples_to_show], encodes[:examples_to_show],
                              encode_decode[:examples_to_show],
                              save_path="result/ae_multi_layers/result-{}-{}-{}-{}.jpg"
                              .format(epoch, n_hidden_1, n_hidden_2, n_hidden_3))

    print("Optimization Finished!")

    # 显示编码结果和解码后结果
    encodes, encode_decode, total_cost = sess.run([encoder_op, y_pred, cost], feed_dict={x: mnist.test.images})
    print(time.strftime("%H:%M:%S", time.localtime()), "Total cost: {}".format(total_cost))

    # 对比原始图片重建图片
    tools.save_result(mnist.test.images[:examples_to_show], encodes[:examples_to_show], encode_decode[:examples_to_show],
                save_path="result/ae_multi_layers/result-{}-{}-{}-{}.jpg"
                      .format(training_epochs, n_hidden_1, n_hidden_2, n_hidden_3))