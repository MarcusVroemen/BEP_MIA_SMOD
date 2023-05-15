import logging
import numpy as np
import tensorflow as tf
import functions.tf_utils as tfu


def network(images, bn_training, detailed_summary=False, use_keras=False):
    common = {'padding': 'valid', 'activation': 'ReLu', 'bn_training': bn_training, 'use_keras': use_keras}
    with tf.variable_scope('AdditiveNoise'):
        images = tf.cond(bn_training,
                         lambda: images + tf.round(tf.random_normal(tf.shape(images),
                                                                    mean=tf.round(tf.random_normal([1], mean=0, stddev=2, dtype=tf.float32)),
                                                                    stddev=2, dtype=tf.float32)),
                         lambda: images)

    with tf.variable_scope('SplitImages'):
        im = images[:, :, :, :, 0, tf.newaxis]
        im_deformed = images[:, :, :, :, 1, tf.newaxis]

    with tf.variable_scope('DownSampling_R4'):
        kernel_bspline_r4 = tfu.kernels.kernel_bspline_r4()
        im_r4 = tf.nn.convolution(im, kernel_bspline_r4, 'VALID', strides=(4, 4, 4))
        im_deformed_r4 = tf.nn.convolution(im_deformed, kernel_bspline_r4, 'VALID', strides=(4, 4, 4))
        images_r4 = tf.concat([im_r4, im_deformed_r4], axis=-1)

    with tf.variable_scope('DownSampling_R2'):
        kernel_bspline_r2 = tfu.kernels.kernel_bspline_r2()
        margin_r2 = 19
        im_r2 = tf.nn.convolution(im[:, margin_r2:-margin_r2, margin_r2:-margin_r2, margin_r2:-margin_r2, :], kernel_bspline_r2, 'VALID', strides=(2, 2, 2))
        im_deformed_r2 = tf.nn.convolution(im_deformed[:, margin_r2:-margin_r2, margin_r2:-margin_r2, margin_r2:-margin_r2, :], kernel_bspline_r2, 'VALID', strides=(2, 2, 2))
        images_r2 = tf.concat([im_r2, im_deformed_r2], axis=-1)

    with tf.variable_scope('DownSampling_R1'):
        margin_r1 = 34
        images_r1 = tf.concat([im[:, margin_r1:-margin_r1, margin_r1:-margin_r1, margin_r1:-margin_r1, :],
                               im_deformed[:, margin_r1:-margin_r1, margin_r1:-margin_r1, margin_r1:-margin_r1, :]],
                              axis=-1)

    with tf.variable_scope('R1'):
        conv1_r1 = tfu.layers.conv3d(images_r1, 16, [3, 3, 3], padding='same', activation='ReLu', bn_training=bn_training, scope='conv1_R1', use_keras=use_keras)
        conv2_r1 = tfu.layers.conv3d(conv1_r1, 20, [3, 3, 3], scope='conv2_R1', **common)
        conv3_r1 = tfu.layers.conv3d(conv2_r1, 24, [3, 3, 3], scope='conv3_R1', **common)
        conv4_r1 = tfu.layers.conv3d(conv3_r1, 28, [3, 3, 3], scope='conv4_R1', **common)
        conv5_r1 = tfu.layers.conv3d(conv4_r1, 32, [3, 3, 3], scope='conv5_R1', **common)
        conv6_r1 = tfu.layers.conv3d(conv5_r1, 32, [3, 3, 3], scope='conv6_R1', **common)
        conv7_r1 = tfu.layers.conv3d(conv6_r1, 32, [3, 3, 3], scope='conv7_R1', **common)

    margin_r2_up = 5
    with tf.variable_scope('R2'):
        conv1_r2 = tfu.layers.conv3d(images_r2, 20, [3, 3, 3], padding='same', activation='ReLu', bn_training=bn_training, scope='conv2_R1', use_keras=use_keras)
        conv2_r2 = tfu.layers.conv3d(conv1_r2, 30, [3, 3, 3], scope='conv2_R2', **common)
        conv3_r2 = tfu.layers.conv3d(conv2_r2, 32, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv3_R2', **common)
        conv4_r2 = tfu.layers.conv3d(conv3_r2, 34, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv4_R2', **common)
        conv5_r2 = tfu.layers.conv3d(conv4_r2, 36, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv5_R2', **common)
        conv6_r2 = tfu.layers.conv3d(conv5_r2, 38, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv6_R2', **common)
        conv7_r2 = tfu.layers.upsampling3d(conv6_r2, scope='conv7_R2', interpolator='trilinear')
        concat_r2 = tf.concat([conv7_r2, conv1_r1[:, margin_r2_up:-margin_r2_up, margin_r2_up:-margin_r2_up, margin_r2_up:-margin_r2_up, :]], axis=-1)
        conv8_r2 = tfu.layers.conv3d(concat_r2, 40, [3, 3, 3], scope='conv8_R2', **common)

    margin_r4_up1 = 1
    margin_r4_up2 = 5
    with tf.variable_scope('R4'):
        conv1_r4 = tfu.layers.conv3d(images_r4, 30, [3, 3, 3], scope='conv1_R4', **common)
        conv2_r4 = tfu.layers.conv3d(conv1_r4, 40, [3, 3, 3], scope='conv2_R4', **common)
        conv3_r4 = tfu.layers.conv3d(conv2_r4, 42, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv3_R4', **common)
        conv4_r4 = tfu.layers.conv3d(conv3_r4, 44, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv4_R4', **common)
        conv5_r4 = tfu.layers.conv3d(conv4_r4, 46, [3, 3, 3], dilation_rate=(2, 2, 2), scope='conv5_R4', **common)
        conv6_r4 = tfu.layers.upsampling3d(conv5_r4, scope='conv6_R4', interpolator='trilinear')
        concat1_r4 = tf.concat([conv6_r4, conv5_r2[:, margin_r4_up1:-margin_r4_up1, margin_r4_up1:-margin_r4_up1, margin_r4_up1:-margin_r4_up1, :]], axis=-1)
        conv7_r4 = tfu.layers.conv3d(concat1_r4, 46, [3, 3, 3], scope='conv7_R4', **common)
        conv8_r4 = tfu.layers.upsampling3d(conv7_r4, scope='conv8_R4', interpolator='trilinear')
        concat2_r4 = tf.concat([conv8_r4, conv1_r1[:, margin_r4_up2:-margin_r4_up2, margin_r4_up2:-margin_r4_up2, margin_r4_up2:-margin_r4_up2, :]], axis=-1)
        conv9_r4 = tfu.layers.conv3d(concat2_r4, 40, [3, 3, 3], scope='conv9_R4', **common)

    with tf.variable_scope('Merged'):
        conv_concat = tf.concat([conv7_r1, conv8_r2, conv9_r4], 4)

    with tf.variable_scope('FullyConnected'):
        conv7 = tfu.layers.conv3d(conv_concat, 120, [3, 3, 3], padding='valid', activation='ELu', bn_training=bn_training, scope='conv1_FC', use_keras=use_keras)
        conv8 = tfu.layers.conv3d(conv7, 50, [3, 3, 3], padding='valid', activation='ELu', bn_training=bn_training, scope='conv2_FC', use_keras=use_keras)

    with tf.variable_scope('DVF'):
        dvf_regnet = tfu.layers.conv3d(conv8, 3, [1, 1, 1], padding="valid", activation=None, bn_training=None, scope='DVF_RegNet', use_keras=use_keras)

    if detailed_summary:
        for i in range(1, 8):
            tensor_name = 'conv'+str(i)+'_R1'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R1_conv'+str(i), selected_slices=1)
        for i in range(1, 9):
            tensor_name = 'conv' + str(i) + '_R2'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R2_conv'+str(i), selected_slices=1)
        for i in range(1, 10):
            tensor_name = 'conv'+str(i)+'_R4'
            tfu.summary.tensor2summary(eval(tensor_name.lower()), tensor_name, scope='DetailedSummaryImages_R4_conv'+str(i), selected_slices=1)
        tfu.summary.tensor2summary(conv7, 'conv7', scope='DetailedSummaryImages_conv7', selected_slices=1)
        tfu.summary.tensor2summary(conv8, 'conv8', scope='DetailedSummaryImages_conv8', selected_slices=1)

    return dvf_regnet


def run_network(train_mode='Train', use_keras=False):
    """
    :param train_mode: 'Training', 'Test'
    :return:
    """
    batch_size = 15
    d_in_val = 101

    if train_mode == 'Train':
        d_in_train, d_out_train = [2 * i + 1 for i in raidus_train()]
        d_in_tf = d_in_train
        d_out_tf = d_out_train
    elif train_mode == 'Test':
        d_in_test, d_out_test = [2 * i + 1 for i in maximum_radius_test()]
        d_in_tf = d_in_test
        d_out_tf = d_out_test
    else:
        raise ValueError('Train or Test')

    import time
    reg_weight = 0.1
    learning_rate = 1E-3
    tf.reset_default_graph()
    tf.set_random_seed(0)
    images_tf = tf.placeholder(tf.float32, shape=[None, d_in_tf, d_in_tf, d_in_tf, 2], name="Images")
    bn_training = tf.placeholder(tf.bool, name='bn_training')
    dvf_ground_truth = tf.placeholder(tf.float32, shape=[None, d_out_tf, d_out_tf, d_out_tf, 3], name="DVF_GroundTruth")
    dvf_predict = network(images_tf, bn_training, detailed_summary=False, use_keras=use_keras)
    huber = (tf.losses.huber_loss(dvf_ground_truth, dvf_predict, weights=1))
    with tf.variable_scope('bending_energy'):
        bending_energy = tfu.image_processing.bending_energy(dvf_predict, voxel_size=[1, 1, 1])
    loss = tf.add(huber, reg_weight * bending_energy)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
    with tf.device('/device:GPU:0'):  # Replace with device you are interested in
        bytes_in_use_tf = BytesInUse()

    if train_mode == 'Test':
        for i in range(10):
            batch_im = np.random.rand(1, d_in_test, d_in_test, d_in_test, 2)
            batch_dvf = np.random.rand(1, d_out_test, d_out_test, d_out_test, 3)
            time_before = time.time()
            byte_in_use, _ = sess.run([bytes_in_use_tf, dvf_predict],
                                      feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 0})
            time_after = time.time()
            logging.info('Testing i={} is done in {:.2f}, byte_in_use={:.1f} MB'.format(i, time_after-time_before, byte_in_use/1024/1024))

    if train_mode == 'Train':
        # training
        for i in range(10):
            batch_im = np.random.rand(batch_size, d_in_train, d_in_train, d_in_train, 2)
            batch_dvf = np.random.rand(batch_size, d_out_train, d_out_train, d_out_train, 3)
            time_before = time.time()
            byte_in_use, _, _, _, _ = sess.run([bytes_in_use_tf, loss, huber, bending_energy, train_step],
                                               feed_dict={images_tf: batch_im, dvf_ground_truth: batch_dvf, bn_training: 1})
            time_after = time.time()
            logging.info('Training i={} is done in {:.2f}, byte_in_use={:.1f} MB'.format(i, time_after-time_before, byte_in_use/1024/1024))

        # validation
        for i in range(5):
            batch_im = np.random.rand(batch_size, d_in_val, d_in_val, d_in_val, 2)
            time_before = time.time()
            sess.run([dvf_predict],
                     feed_dict={images_tf: batch_im, bn_training: 0})
            time_after = time.time()
            logging.info('Validation i={} is done in {:.2f}'.format(i, time_after-time_before))


def raidus_train():
    """
    please note that the side of the patch is equal to 2*r_input+1
    :return:
    """
    r_input = 52
    r_output = 10
    return r_input, r_output


def maximum_radius_test(gpu_memory=None, number_of_gpu=None):
    """
    :return:
    """
    if gpu_memory is None and number_of_gpu is None:
        gpu_memory, number_of_gpu = tfu.client.read_gpu_memory()
    logging.info('GPU Memory={:.2f} Number of GPU={}'.format(gpu_memory, number_of_gpu))
    if 10 <= gpu_memory < 11:
        r_input = 120
        r_output = 78
    elif 11 <= gpu_memory < 12:
        r_input = '!'
        r_output = '!'
    elif 12 <= gpu_memory < 16:
        r_input = 136
        r_output = 94
    return r_input, r_output


def get_resize_unit():
    """
    Accepted value that you can add or subtract to the network radius.
    This is relevant to the number of max-pooling insided the network
    :return: resize_unit
    """
    resize_unit = 4
    return resize_unit


def get_padto():
    padto = None
    return padto


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    d_input, d_output = [2 * i + 1 for i in raidus_train()]
    use_keras = False
    print_all = True
    do_training = True

    network(tf.placeholder(tf.float32, shape=[None, d_input, d_input, d_input, 2]),
            tf.placeholder(tf.bool, name='bn_training'), use_keras=use_keras)
    logging.info('total number of variables %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    if print_all:
        # gr = tf.get_default_graph()
        # for op in tf.get_default_graph().get_operations():
        #     logging.info(str(op.name))
        logging.info('printint trainables \n---------------\n--------------\n-------------')
        for v in tf.trainable_variables():
            logging.info(v)

    if do_training:
        run_network(train_mode='Test', use_keras=use_keras)
