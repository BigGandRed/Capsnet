from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
from utils import _conv2d_wrapper
from layer import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer
import tensorflow.contrib.slim as slim

def baseline_model_cnn(X, num_classes):
    nets = _conv2d_wrapper(
        X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
        add_bias=False, activation_fn=tf.nn.relu, name='conv1'
        )      #   使用一个卷积层对输入X进行处理,卷积核的大小为3x300
    nets = slim.flatten(nets)   #将卷积层的输出展平（flatten）。这是为了将二维的特征映射转换为一维，以便它们可以被全连接层处理。
    tf.logging.info('flatten shape: {}'.format(nets.get_shape()))
    nets = slim.fully_connected(nets, 128, scope='relu_fc3', activation_fn=tf.nn.relu) #展平后的数据被送入一个全连接层，这个层有128个神经元
    tf.logging.info('fc shape: {}'.format(nets.get_shape()))
    
    activations = tf.sigmoid(slim.fully_connected(nets, num_classes, scope='final_layer', activation_fn=None)) 
    #数据通过另一个全连接层，这个层的神经元数量等于类别数（num_classes），并且不使用激活函数, 输出通过sigmoid函数转换，得到每个类别的预测概率
    tf.logging.info('fc shape: {}'.format(activations.get_shape()))
    return tf.zeros([0]), activations

def baseline_model_kimcnn(X, max_sent, num_classes):
    pooled_outputs = []
    for i, filter_size in enumerate([3,4,5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):            
            filter_shape = [filter_size, 300, 1, 100]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W") #创建了一个具有特定形状和初始化为截断正态分布的权重矩阵
            #tf.truncated_normal 函数用于生成一个截断的正态分布随机数,有助于防止在训练初期出现过大的权重值，从而避免神经网络训练过程中的梯度问题
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")  #偏置变量 b，用于卷积层，初始值设为0.1，100个这样的偏置值对应于卷积层中的100个过滤器
            #tf.constant(0.1, shape=[100]) 表示创建一个形状为 [100]（即包含100个元素的一维数组）的常量，每个元素的初始值都是 0.1
            conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")   #X为输入，W为权重         
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 
            """tf.nn.bias_add(conv, b): 这个函数将偏置b添加到卷积的输出conv上。具体来说,它会对每个卷积输出的特征图(feature map)添加对应的偏置值。
               这个操作是逐元素进行的,即每个特征图的每个元素都会加上相应的偏置值。并通过ReLU函数进行激活。
            """            
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sent - filter_size + 1, 1, 1],  #ksize: 池化窗口的大小
                strides=[1, 1, 1, 1],  #意味着池化窗口在所有方向上的滑动步长都是1。
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)  #将pooled添加到pooled_outputs列表中。这个列表最终会包含不同卷积核大小下的所有池化输出
    num_filters_total = 100 * 3  #模型对每种大小的卷积核（3, 4, 5）都使用了100个过滤器
    h_pool = tf.concat(pooled_outputs, 3)  #tf.concat 函数在第四个维度（索引为3）上合并三个不同大小的卷积核产生的池化输出
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    activations = tf.sigmoid(slim.fully_connected(h_pool_flat, num_classes, scope='final_layer', activation_fn=None))
    return tf.zeros([0]), activations
        
def capsule_model_B(X, num_classes):
    poses_list = []
    for _, ngram in enumerate([3,4,5]):   #enumerate 函数用于同时获取列表中的元素及其对应的索引
        with tf.variable_scope('capsule_'+str(ngram)): 
            nets = _conv2d_wrapper(
                X, shape=[ngram, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
            tf.logging.info('output shape: {}'.format(nets.get_shape()))
            nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                                 padding='VALID', pose_shape=16, add_bias=True, name='primary')  #初始化胶囊层，定义胶囊的大小和数量                      
            nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2') #使用胶囊卷积层进一步处理数据
            nets = capsule_flatten(nets)  #将胶囊卷积层的输出展平
            poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
            poses_list.append(poses)
    
    poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0) 
    activations = K.sqrt(K.sum(K.square(poses), 2))
    return poses, activations

def capsule_model_A(X, num_classes):
    with tf.variable_scope('capsule_'+str(3)):   
        nets = _conv2d_wrapper(
                X, shape=[3, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
        nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2') 
    return poses, activations