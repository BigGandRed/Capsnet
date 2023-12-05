import tensorflow as tf
import keras
from keras import backend as K
from utils import _conv2d_wrapper, _get_weights_wrapper

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))  #x减去x在指定轴 axis上的最大值，然后计算 e 的指数,保持输出的维度与输入相同
    return ex/K.sum(ex, axis=axis, keepdims=True)  #计算每个元素的指数，然后除以它们在指定轴上的总和，得到 softmax 的输出。使输出的形状与输入一致

def squash_v1(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    #首先计算 x 的平方和,方向和维度略，再加上一个非常小的数（K.epsilon()）以提高数值稳定性
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)  #计算一个缩放因子，sqrt计算平方根。这个公式保证了输出向量的长度在0到1之间
    return scale * x

def squash_v0(s, axis=-1, epsilon=1e-7, name=None):
    s_squared_norm = K.sum(K.square(s), axis, keepdims=True) + K.epsilon()  #与 squash_v1 类似，首先计算 s 的平方和加上一个小常数
    safe_norm = K.sqrt(s_squared_norm)
    scale = 1 - tf.exp(-safe_norm)  #使用不同的方法来计算缩放因子。这里使用了 safe_norm（向量长度的平方根）的指数衰减函数来计算缩放因子
    return scale * s / safe_norm   #使得长向量被缩短，而短向量长度增加
   
def routing(u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations):   #用于计算胶囊之间的动态路由
    """ u_hat_vecs: 输入向量，通常是由前一层的胶囊输出经过变换得到的预测向量;beta_a: 用于路由过程的偏置参数
        iterations: 动态路由的迭代次数;output_capsule_num: 输出胶囊的数量;i_activations: 前一层胶囊的激活值
    """
    b = keras.backend.zeros_like(u_hat_vecs[:,:,:,0]) #keras.backend.zeros_like 函数用于创建一个与给定张量形状相同的零张量
    #它根据 u_hat_vecs[:,:,:,0] 的形状创建一个新的零张量 b。这意味着 b 将与 u_hat_vecs 的前三个维度形状相同，但不包含第四个维度
    #在胶囊网络的动态路由过程中，b 代表了初步的“投票”权重。这些权重用于确定不同输入胶囊对输出胶囊的影响大小
    if i_activations is not None:
        i_activations = i_activations[...,tf.newaxis] #如果提供了激活值 i_activations，则在其最后增加一个新的维度
    for i in range(iterations):
        if False:
            leak = tf.zeros_like(b, optimize=True)
            leak = tf.reduce_sum(leak, axis=1, keep_dims=True)
            leaky_logits = tf.concat([leak, b], axis=1)
            leaky_routing = tf.nn.softmax(leaky_logits, dim=1)        
            c = tf.split(leaky_routing, [1, output_capsule_num], axis=1)[1]   # 这里的代码块实际上是不会执行的，因为条件永远为 False。
        else:                                                                 # 看起来像是一个占位符，可能用于未来的扩展或备用逻辑。
            c = softmax(b, 1)   #这一步是动态路由过程的关键，它决定了每个输入胶囊对输出胶囊的贡献程度。
#        if i_activations is not None:
#            tf.transpose(tf.transpose(c, perm=[0,2,1]) * i_activations, perm=[0,2,1]) 
        outputs = squash_v1(K.batch_dot(c, u_hat_vecs, [2, 2]))   #首先计算 c 和 u_hat_vecs 的点积，然后通过 squash_v1 函数进行激活和压缩。
        if i < iterations - 1:
            b = b + K.batch_dot(outputs, u_hat_vecs, [2, 3])      #在每次迭代结束时，根据输出胶囊的预测向量和输入向量更新路由权重 b。                              
    poses = outputs 
    activations = K.sqrt(K.sum(K.square(poses), 2))
    return poses, activations


def vec_transformationByConv(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num):  #poses: 输入胶囊的姿势向量                          
    """ vec_transformationByConv 函数通过一维卷积操作和权重变换处理输入胶囊的姿势向量，生成输出胶囊的预测向量。
        这是胶囊网络中实现复杂特征转换的关键步骤，它允许网络学习从一个层级的特征到另一个层级特征的转换关系。
    """
    kernel = _get_weights_wrapper(
      name='weights', shape=[1, input_capsule_dim, output_capsule_dim*output_capsule_num], weights_decay_factor=0.0
    )   # 调用自utils.py,创建权重变量（kernel）,由方法名可知
    tf.logging.info('poses: {}'.format(poses.get_shape()))   
    tf.logging.info('kernel: {}'.format(kernel.get_shape()))
    u_hat_vecs = keras.backend.conv1d(poses, kernel)
    """使用 keras.backend.conv1d 函数执行一维卷积操作。这里的卷积不是传统意义上对图像的卷积，而是将权重 kernel 应用于姿势向量 poses,
       以产生转换后的向量 u_hat_vecs。这个过程模仿了神经网络中的全连接层,但以一种适合胶囊网络的方式进行。
    """
    u_hat_vecs = keras.backend.reshape(u_hat_vecs, (-1, input_capsule_num, output_capsule_num, output_capsule_dim))
    u_hat_vecs = keras.backend.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))  #permute_dimensions 调整维度的顺序
    return u_hat_vecs

def vec_transformationByMat(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num, shared=True):                        
    """vec_transformationByMat 函数在胶囊网络中负责将输入胶囊层的信息通过一定的变换矩阵映射到输出胶囊层
       shared: 表示是否在不同的输入胶囊间共享变换矩阵
    """
    inputs_poses_shape = poses.get_shape().as_list()
    poses = poses[..., tf.newaxis, :]        
    # ...: 在 Python 中用于表示选择数组的所有现有维度。poses[..., tf.newaxis, :] 的作用是在 poses 数组的最后一个维度之前增加一个新的维度
    poses = tf.tile(
              poses, [1, 1, output_capsule_num, 1]
            )  #将输入胶囊的姿态矩阵 poses 沿着第三个维度（输出胶囊数量维度)复制output_capsule_num次。可以将输入胶囊的信息传递给多个输出胶囊,1表示不复制
    if shared:
        kernel = _get_weights_wrapper(
          name='weights', shape=[1, 1, output_capsule_num, output_capsule_dim, input_capsule_dim], weights_decay_factor=0.0
        )
        kernel = tf.tile(
                  kernel, [inputs_poses_shape[0], input_capsule_num, 1, 1, 1]
                )
    else:
        kernel = _get_weights_wrapper(
          name='weights', shape=[1, input_capsule_num, output_capsule_num, output_capsule_dim, input_capsule_dim], weights_decay_factor=0.0
        )   #为每个输入胶囊创建一个单独的变换矩阵
        kernel = tf.tile(
                  kernel, [inputs_poses_shape[0], 1, 1, 1, 1]
                )
    tf.logging.info('poses: {}'.format(poses[...,tf.newaxis].get_shape()))   
    tf.logging.info('kernel: {}'.format(kernel.get_shape()))
    u_hat_vecs = tf.squeeze(tf.matmul(kernel, poses[...,tf.newaxis]),axis=-1) 
    #使用 tf.matmul计算矩阵乘法,tf.squeeze 用于移除结果中长度为1的维度，在这个例子中，axis=-1 指定了只移除最后一个维度（如果它的大小为1）。
    u_hat_vecs = keras.backend.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
    return u_hat_vecs

def capsules_init(inputs, shape, strides, padding, pose_shape, add_bias, name):
    """capsules_init 函数通过一系列变换和操作，从输入数据中创建了胶囊网络的第一层。
       这包括应用卷积操作、调整维度、应用激活函数，并计算每个胶囊的激活值。
       inputs: 网络的输入张量;shape: 卷积层的过滤器（或称为核）的尺寸和数量。
    """
    with tf.variable_scope(name):   
        poses = _conv2d_wrapper(
          inputs,         
          shape=shape[0:-1] + [shape[-1] * pose_shape],  #计算卷积核的新尺寸，以适应胶囊的姿态矩阵大小
          strides=strides,
          padding=padding,
          add_bias=add_bias,
          activation_fn=None,
          name='pose_stacked'
        )        #创建一个卷积层
        poses_shape = poses.get_shape().as_list()    
        poses = tf.reshape(
                    poses, [
                        -1, poses_shape[1], poses_shape[2], shape[-1], pose_shape
                    ])        #poses 被重塑成一个五维张量，以适应胶囊的结构
        beta_a = _get_weights_wrapper(
                        name='beta_a', shape=[1, shape[-1]]
                    )    #beta_a 用于初始化胶囊层,针对初始化层中的胶囊的激活值
        poses = squash_v1(poses, axis=-1)  #应用 squash_v1 函数对 poses 进行激活和压缩，使输出的姿态向量长度在 0 到 1 之间
        activations = K.sqrt(K.sum(K.square(poses), axis=-1)) + beta_a     #计算每个胶囊的激活值，即姿态向量的长度，然后加上偏置 beta_a   
        tf.logging.info("prim poses dimension:{}".format(poses.get_shape()))

    return poses, activations

def capsule_fc_layer(nets, output_capsule_num, iterations, name):
    """ nets: 输入的胶囊网络层，通常包括姿态 和 激活值;name: 这个胶囊层的名称
        它首先通过姿态变换获取预测向量，然后通过动态路由算法来更新胶囊的姿态和激活值，最终输出到下一层或作为网络的最终输出。
    """ 
    with tf.variable_scope(name):   
        poses, i_activations = nets
        input_pose_shape = poses.get_shape().as_list()

        u_hat_vecs = vec_transformationByConv(
                      poses,
                      input_pose_shape[-1], input_pose_shape[1],
                      input_pose_shape[-1], output_capsule_num,
                      )  #调用前面的函数通过卷积操作进行姿态变换
        
        tf.logging.info('votes shape: {}'.format(u_hat_vecs.get_shape()))
        
        beta_a = _get_weights_wrapper(
                name='beta_a', shape=[1, output_capsule_num]
                )
      #由维度可知，每个输出胶囊都有一个对应的 beta_a 值，这些值用于调整或"偏置"该胶囊的激活值
      
        poses, activations = routing(u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations)
        
        tf.logging.info('capsule fc shape: {}'.format(poses.get_shape()))   
        
    return poses, activations

def capsule_flatten(nets):
  """ capsule_flatten 函数的作用是将胶囊网络中的胶囊层输出平铺(flatten),以便能够将这些输出作为后续层(比如全连接层)的输入。
  """
    poses, activations = nets
    input_pose_shape = poses.get_shape().as_list()
    
    poses = tf.reshape(poses, [
                    -1, input_pose_shape[1]*input_pose_shape[2]*input_pose_shape[3], input_pose_shape[-1]]) 
    activations = tf.reshape(activations, [
                    -1, input_pose_shape[1]*input_pose_shape[2]*input_pose_shape[3]])
    tf.logging.info("flatten poses dimension:{}".format(poses.get_shape()))
    tf.logging.info("flatten activations dimension:{}".format(activations.get_shape()))

    return poses, activations

def capsule_conv_layer(nets, shape, strides, iterations, name):   
    """ capsule_conv_layer 函数是胶囊网络中的一个重要组成部分，它实现了一种卷积胶囊层。
        这个层通过对输入胶囊的姿态进行卷积变换和动态路由，计算出输出胶囊的姿态和激活值，从而捕捉更高层次的特征表示。
    """
    with tf.variable_scope(name):              
        poses, i_activations = nets
        
        inputs_poses_shape = poses.get_shape().as_list()

        hk_offsets = [
          [(h_offset + k_offset) for k_offset in range(0, shape[0])] for h_offset in
          range(0, inputs_poses_shape[1] + 1 - shape[0], strides[1])
        ]
        wk_offsets = [
          [(w_offset + k_offset) for k_offset in range(0, shape[1])] for w_offset in
          range(0, inputs_poses_shape[2] + 1 - shape[1], strides[2])
        ]
    """这部分代码生成了卷积核在每个维度上的偏移量列表。这些偏移量用于提取输入姿态(poses)的局部区域,类似于传统卷积操作中卷积核滑动的方式.
    """
        inputs_poses_patches = tf.transpose(    #使用 tf.transpose 重排这些提取出的姿态的维度
          tf.gather(
            tf.gather(     ##沿着 axis=1（高度方向）在 poses 上收集由 hk_offsets 指定的序列。hk_offsets 包含了卷积核在高度方向上应用的偏移量
              poses, hk_offsets, axis=1, name='gather_poses_height_kernel' 
            ), wk_offsets, axis=3, name='gather_poses_width_kernel'   #wk_offsets 指定了卷积核在宽度方向上的偏移量
          ), perm=[0, 1, 3, 2, 4, 5, 6], name='inputs_poses_patches'  
        )
        tf.logging.info('i_poses_patches shape: {}'.format(inputs_poses_patches.get_shape()))
    
        inputs_poses_shape = inputs_poses_patches.get_shape().as_list()
        inputs_poses_patches = tf.reshape(inputs_poses_patches, [
                                -1, shape[0]*shape[1]*shape[2], inputs_poses_shape[-1]
                                ])

        i_activations_patches = tf.transpose(
          tf.gather(
            tf.gather(
              i_activations, hk_offsets, axis=1, name='gather_activations_height_kernel'
            ), wk_offsets, axis=3, name='gather_activations_width_kernel'
          ), perm=[0, 1, 3, 2, 4, 5], name='inputs_activations_patches'
        )
    """利用偏移量列表和 tf.gather 函数从输入姿态(poses)和激活值中提取局部区域(patches)。这些局部区域随后将被用于胶囊的动态路由
    """
        tf.logging.info('i_activations_patches shape: {}'.format(i_activations_patches.get_shape()))
        i_activations_patches = tf.reshape(i_activations_patches, [
                                -1, shape[0]*shape[1]*shape[2]]
                                )
        u_hat_vecs = vec_transformationByConv(
                  inputs_poses_patches,
                  inputs_poses_shape[-1], shape[0]*shape[1]*shape[2],
                  inputs_poses_shape[-1], shape[3],
                  )  #使用前面定义的函数对姿态 poses行变换。这个步骤类似于卷积操作，但是针对的是胶囊网络中的姿态（poses）
        tf.logging.info('capsule conv votes shape: {}'.format(u_hat_vecs.get_shape()))
    
        beta_a = _get_weights_wrapper(
                name='beta_a', shape=[1, shape[3]]
                ) #创建路由所需的偏置参数 beta_a
        poses, activations = routing(u_hat_vecs, beta_a, iterations, shape[3], i_activations_patches)
        poses = tf.reshape(poses, [
                    inputs_poses_shape[0], inputs_poses_shape[1],
                    inputs_poses_shape[2], shape[3],
                    inputs_poses_shape[-1]]
                ) 
        activations = tf.reshape(activations, [
                    inputs_poses_shape[0],inputs_poses_shape[1],
                    inputs_poses_shape[2],shape[3]]
                ) 
        nets = poses, activations            
    tf.logging.info("capsule conv poses dimension:{}".format(poses.get_shape()))
    tf.logging.info("capsule conv activations dimension:{}".format(activations.get_shape()))
    return nets