'''
自编码使用的一些工具
'''
import sklearn.preprocessing as prep
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# 数据预算处理, [0, 1]
def minmax_scale(x_train, x_test):
    preprocesser = prep.MinMaxScaler().fit(x_train)
    return preprocesser.transform(x_train), preprocesser.transform(x_test)
    pass

# xavier 方式数据初始化
def xavier_init(fan_in, fan_out, constant = 1):
    low = - constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval= high, dtype= tf.float32)


# 获取随机block数据
def get_random_block_from_data(data, batch_size, fixed = False):
    start_index = 0 if fixed else np.random.randint(0, len(data) - batch_size)
    return data[start_index : start_index + batch_size]

# 打印中间结果
def save_result(images, encodes, decode, n_show=10, save_path="result/result.jpg"):
    path, _ = os.path.split(save_path)
    if not os.path.exists(path):
        os.makedirs(path)

    # 对比原始图片重构图片
    plt.figure(figsize=(n_show, 3))
    gs = gridspec.GridSpec(3, n_show)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(n_show):
        # 原始图片
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(images[i], (28, 28)))

        # 编码后的图
        ax = plt.subplot(gs[i + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(encodes[i], (encodes[i].shape[0]//8, 8)))

        # 解码后的图
        ax = plt.subplot(gs[i + n_show + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(decode[i], (28, 28)))
        pass

    plt.savefig("{}.png".format(os.path.splitext(save_path)[0]), bbox_inches='tight')
    plt.close()
    pass

# 打印中间结果，编码打印散点图
def save_result_scatter(images, encodes, decode, labels, n_show=10, save_path="result/result.jpg"):
    path, _ = os.path.split(save_path)
    if not os.path.exists(path):
        os.makedirs(path)

    # 对比原始图片重构图片
    plt.figure(figsize=(n_show, 3))
    gs = gridspec.GridSpec(3, n_show)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(n_show):
        # 原始图片
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(images[i], (28, 28)))

        # 编码后的图
        ax = plt.subplot(gs[i + n_show])
        ax.scatter(encodes[:, 0], encodes[:, 1], c=labels)
        ax.colorbar()

        # 解码后的图
        ax = plt.subplot(gs[i + n_show + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(decode[i], (28, 28)))
        pass

    plt.savefig("{}.png".format(os.path.splitext(save_path)[0]), bbox_inches='tight')
    plt.close()
    pass

# 打印中间结果
def gaussian_save_result(images, gaussian_images, encode_decode, n_show=10, save_path="result/result.jpg"):

    # 创建文件夹
    path, _ = os.path.split(save_path)
    if not os.path.exists(path):
        os.makedirs(path)

    # 对比原始图片重建图片
    plt.figure(figsize=(n_show, 3))
    gs = gridspec.GridSpec(3, n_show)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(n_show):
        # 原始图片
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(images[i], (28, 28)))

        # 编码后的图
        ax = plt.subplot(gs[i + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(gaussian_images[i], (28, 28)))

        # 解码后的图
        ax = plt.subplot(gs[i + n_show + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.savefig("{}.png".format(os.path.splitext(save_path)[0]), bbox_inches='tight')
    plt.close()
    pass