from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.layers.python.layers import initializers

import tensorflow as tf

slim = tf.contrib.slim   #定义 slim 作为 TensorFlow 的一个便捷接口

epsilon = 1e-9    #变量定义了一个非常小的数，通常用于避免除零错误或进行数值稳定性操作

def _matmul_broadcast(x, y, name):  #x 和 y 是将要进行矩阵乘法的两个输入张量，name是TensorFlow 计算图中的变量作用域
  """Compute x @ y, broadcasting over the first `N - 2` ranks.
  """
  with tf.variable_scope(name) as scope:   #创建了一个 TensorFlow 变量作用域，命名为提供的 name
    return tf.reduce_sum(        #使用了广播机制来扩展 x 和 y 的维度，tf.newaxis 用于在指定的轴上增加一个新维度
      tf.nn.dropout(x[..., tf.newaxis] * y[..., tf.newaxis, :, :],1), axis=-2   #应用 dropout 正则化。但dropout 保留率设置为 1 (rate=1)
    )    #使用 tf.reduce_sum 沿倒数第二个轴（axis=-2）对结果进行求和


def _get_variable_wrapper(
  name, shape=None, dtype=None, initializer=None,    #变量的名称、形状、类型和初始化器都默认为none。
  regularizer=None,   #变量的正则化函数，默认为 None
  trainable=True,    #是否可训练，默认为 True
  collections=None,   #变量应该被添加到的 TensorFlow 集合列表，默认为 None
  caching_device=None,   #缓存设备，默认为 None
  partitioner=None,     #分区器，默认为 None
  validate_shape=True,   #是否验证形状，默认为 True
  custom_getter=None     #自定义获取函数，默认为 None
):
  """Wrapper over tf.get_variable().
  """

  with tf.device('/cpu:0'):   #这行代码指定了变量应该被存储在 CPU 上。'/cpu:0' 表示第一个 CPU 设备
    var = tf.get_variable(    
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, trainable=trainable,
      collections=collections, caching_device=caching_device,
      partitioner=partitioner, validate_shape=validate_shape,
      custom_getter=custom_getter
    )      #调用 tf.get_variable 来创建一个新变量或返回一个已存在的变量
  return var


def _get_weights_wrapper(
  name, shape, dtype=tf.float32, initializer=initializers.xavier_initializer(),   #常用的权重初始化方法，用于帮助神经网络的训练开始时更加稳定
  weights_decay_factor=None  #权重衰减因子，用于正则化，可以帮助防止模型过拟合
):
  """Wrapper over _get_variable_wrapper() to get weights, with weights decay factor in loss.
  """

  weights = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )   #其属性（如名称、形状、数据类型、初始化方法）由 _get_weights_wrapper 函数的参数确定

  if weights_decay_factor is not None and weights_decay_factor > 0.0:

    weights_wd = tf.multiply(
      tf.nn.l2_loss(weights), weights_decay_factor, name=name + '/l2loss'  #它将这个操作的名称设置为原来权重的名称后面加上 '/l2loss'
    )      #计算 L2 损失的函数,然后将 L2 损失与 权重衰减因子 相乘

    tf.add_to_collection('losses', weights_wd)   #将计算得到的权重衰减损失加入到 TensorFlow 的 losses 集合中

  return weights


def _get_biases_wrapper(
  name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0)  #偏置初始时被设置为0
):
  """Wrapper over _get_variable_wrapper() to get bias.
  """

  biases = _get_variable_wrapper(   #创建或获取一个偏置变量
    name=name, shape=shape, dtype=dtype, initializer=initializer  #其属性由 _get_biases_wrapper 函数的参数确定
  )
#用于创建或获取神经网络中偏置变量的函数。它使用 _get_variable_wrapper 来实际创建或获取这些偏置变量，并使用提供的参数来定义这些变量
  return biases


def _conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name, stddev=0.1):
  """Wrapper over tf.nn.conv2d().   #add_bias: 布尔值,表示是否在卷积后添加偏置项;activation_fn: 激活函数，用于卷积后的非线性激活
                                    #name: 函数作用域的名称 ;stddev: 权重初始化时的标准差,默认值为0.1
  """

  with tf.variable_scope(name) as scope:
    kernel = _get_weights_wrapper(
      name='weights', shape=shape, weights_decay_factor=0.0, #initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    )   #不使用权重衰减
    output = tf.nn.conv2d(
      inputs, filter=kernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'       
      )          #首先使用 _get_biases_wrapper 创建偏置变量，然后使用 tf.add 将偏置加到卷积的输出上
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )   #如果提供了激活函数 activation_fn，则在卷积的输出上应用这个激活函数

  return output


def _separable_conv2d_wrapper(inputs, depthwise_shape, pointwise_shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.separable_conv2d().  #depthwise_shape: 深度卷积核的形状;pointwise_shape: 逐点卷积核(1x1 卷积)的形状
  """
  
  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=depthwise_shape, weights_decay_factor=0.0
    )             #创建深度卷积核的权重
    pkernel = _get_weights_wrapper(
      name='pointwise_weights', shape=pointwise_shape, weights_decay_factor=0.0
    )            #创建了逐点卷积核的权重
    output = tf.nn.separable_conv2d(
      input=inputs, depthwise_filter=dkernel, pointwise_filter=pkernel,
      strides=strides, padding=padding, name='conv'
    )            #执行可分离卷积操作。深度卷积使用 dkernel 权重，逐点卷积使用 pkernel 权重
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[pointwise_shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _depthwise_conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.depthwise_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=shape, weights_decay_factor=0.0
    )
    output = tf.nn.depthwise_conv2d(
      inputs, filter=dkernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      d_ = output.get_shape()[-1].value
      biases = _get_biases_wrapper(
        name='biases', shape=[d_]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

    return output