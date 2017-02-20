

from scipy.misc import imread, imshow, imresize, imsave
import os
from glob import glob
import numpy as np
# this for sklearn 0.17, for 0.18: use sklearn.model_selection
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
import pdb

def print_shape(t):
    print(t.name, t.get_shape().as_list())

# def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
#     reader = tf.train.NewCheckpointReader(save_file)
#     saved_shapes = reader.get_variable_to_shape_map()
#     var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
#             if var.name.split(':')[0] in saved_shapes])    
#     restore_vars = []    
#     for var_name, saved_var_name in var_names:            
#         curr_var = graph.get_tensor_by_name(var_name)
#         var_shape = curr_var.get_shape().as_list()
#         if var_shape == saved_shapes[saved_var_name]:
#             restore_vars.append(curr_var)
#     opt_saver = tf.train.Saver(restore_vars)
#     opt_saver.restore(session, save_file)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# def residual(inputs, name="res"):
#     with tf.variable_scope(name) as scope:
#         with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, activation_fn=None):

#             net = slim.conv2d(inputs, 64, [3, 3])
#             net = slim.conv2d(net, 64, [3, 3], activation_fn=None)
#             net = net + inputs
#             return net

# create scratch variable scope to get around reuse=True issues with
# temporary variables
# (hack around https://github.com/tensorflow/tensorflow/issues/5827)
with tf.variable_scope("scratch", reuse=False) as scratch_varscope:
    SCRATCH_VARSCOPE = scratch_varscope
assert SCRATCH_VARSCOPE == scratch_varscope
assert SCRATCH_VARSCOPE.reuse == False

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                with tf.variable_scope(SCRATCH_VARSCOPE) as scratch_varscope:
                    assert scratch_varscope.reuse == False
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                    ema_apply_op = self.ema.apply([batch_mean, batch_var])
                    self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
                    with tf.control_dependencies([ema_apply_op]):
                        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

# this is really bad
batch_norm_list = []
nb_residual = 10
n_extra_bn = 1
for n in range(nb_residual*2 + n_extra_bn):
    batch_norm_list.append(batch_norm(name='bn'+str(n)))



########
# CONFIG 
#########
adam_learning_rate = 0.0001
adam_beta1 = 0.9

batch_size = 32
image_h = 128
image_w = 128

num_epochs = 20

###############################
# BUILDING THE MODEL
###############################


real_ims = tf.placeholder(tf.float32, [None, image_h, image_w, 3], name='real_ims')

# the input to the generator is a downsampled version of the real image
inputs = tf.placeholder(tf.float32, [None, image_h/4, image_w/4, 3], name='inputs')


# generator section
print "GENERATOR"
print "-----------"

def create_generator(inputs, b_training=True):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        normalizer_params={'scale':True, 'decay': 0.9, 'updates_collections': None},
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = inputs
        print_shape(net)

        net = tf.nn.relu(slim.conv2d(net, 64, [3, 3], scope='conv1'))
        print_shape(net)

        net1 = net

        res_inputs = net1
        for n in range(nb_residual):
            net = tf.nn.relu(batch_norm_list[n*2](slim.conv2d(res_inputs, 64, [3, 3], scope='conv1_res'+str(n)), train=b_training))
            net = batch_norm_list[n*2+1](slim.conv2d(net, 64, [3, 3], scope='conv2_res'+str(n)), train=b_training)
            net = net + res_inputs
            res_inputs = net


        print_shape(net)

        net = batch_norm_list[-1](slim.conv2d(net, 64, [3, 3], scope='conv2'), train=b_training) + net1
        print_shape(net)

        # deconv
        net = tf.nn.relu(slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv1'))
        print_shape(net)

        net = tf.nn.relu(slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv2'))
        print_shape(net)


        # tanh since images have range [-1,1]
        net = slim.conv2d(net, 3, [3, 3], scope='conv3', activation_fn=tf.nn.tanh)
        print_shape(net)

        return net

with tf.variable_scope("generator") as scope:
    gen = create_generator(inputs)
    scope.reuse_variables()
    gen_test = create_generator(inputs, False)


print "DISCRIMINATOR"
print "--------------"
#disc_bn1 = batch_norm(name='d_bn1')
disc_bn2 = batch_norm(name='d_bn2')
disc_bn3 = batch_norm(name='d_bn3')
disc_bn4 = batch_norm(name='d_bn4')
disc_bn5 = batch_norm(name='d_bn5')
def create_discriminator(inputs):
    with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    activation_fn=None,
                    # activation_fn=lrelu,
                    # normalizer_fn=slim.batch_norm,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    stride=2):

        noisy_inputs = inputs + tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=0.05, dtype=tf.float32)

        # disc = slim.conv2d(noisy_inputs, 64, [5, 5], scope='conv1')
        # print_shape(disc)

        # disc = slim.conv2d(disc, 128, [5, 5], scope='conv2')
        # print_shape(disc)

        # disc = slim.conv2d(disc, 256, [5, 5], scope='conv3')
        # print_shape(disc)

        # disc = slim.conv2d(disc, 512, [5, 5], scope='conv4')
        # print_shape(disc)

        # disc = slim.conv2d(disc, 512, [5, 5], scope='conv5')
        # print_shape(disc)
        disc = lrelu(slim.conv2d(noisy_inputs, 64, [5, 5], scope='conv1'))
        print_shape(disc)

        disc = lrelu(disc_bn2(slim.conv2d(disc, 128, [5, 5], scope='conv2')))
        print_shape(disc)

        disc = lrelu(disc_bn3(slim.conv2d(disc, 256, [5, 5], scope='conv3')))
        print_shape(disc)

        disc = lrelu(disc_bn4(slim.conv2d(disc, 512, [5, 5], scope='conv4')))
        print_shape(disc)

        disc = lrelu(disc_bn5(slim.conv2d(disc, 512, [5, 5], scope='conv5')))
        print_shape(disc)

    disc_logits = slim.fully_connected(tf.contrib.layers.flatten(disc), 1, activation_fn=None, scope='fc6')
    return disc_logits
# create 2 discriminators: for fake and real images
with tf.variable_scope("discriminators") as scope:
    disc_real = create_discriminator(real_ims)
    scope.reuse_variables()
    disc_fake = create_discriminator(gen)


# Losses
###########

# loss on real input images; all outputs should be 1

# make labels noisy
if np.random.rand() < 0.8:
    labels_real = tf.ones_like(disc_real)
    labels_fake = tf.zeros_like(disc_fake)
else:
    labels_real = tf.zeros_like(disc_real)
    labels_fake = tf.ones_like(disc_fake)

d_loss_real = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_real, labels_real))

# loss on fake input images, fakes should be 0
d_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, labels_fake))

# similar to above, but we want fake (generator) images to output 1
g_loss_adv = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))


# L2 LOSS

# pretend we don't have ground truth so all we can do is 
# do reconstruction loss on the input
outputs = tf.image.resize_images(gen, [image_h/4, image_w/4])

diff_mse = tf.contrib.layers.flatten(outputs - inputs)
sum_mse = tf.reduce_sum(tf.square(diff_mse), 1)
g_loss_L2 = tf.reduce_mean(sum_mse/(3*(image_h/4)*(image_h/4)))

g_loss = g_loss_L2 + 0.05*g_loss_adv

d_loss = d_loss_real + d_loss_fake



train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminators' in var.name]
g_vars = [var for var in train_vars if 'generator' in var.name]


# optimize the generator and discriminator separately
d_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(g_loss, var_list=g_vars)
    
weight_saver = tf.train.Saver(max_to_keep=1)

print "initialization done"


#############
# TRAINING
############

data_dir = '/home/wseto/datasets/img_align_celeba128'
#data_dir = '/home/wseto/datasets/wiki_align'
data = glob(os.path.join(data_dir, "*.png"))
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)


print "TRAINING"
print "-----------"

start_time = time.time()
counter = 0

b_load = False

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    for epoch in range(num_epochs):

        np.random.shuffle(data_train)

        if b_load:
            ckpt = tf.train.get_checkpoint_state('/home/wseto/dcgan/checkpoint')
            weight_saver.restore(sess, ckpt.model_checkpoint_path)
            counter = int(ckpt.model_checkpoint_path.split('-', 1)[1]) 
            print "successfuly restored!" + " counter:", counter

        for idx in xrange(num_batches):
            batch_filenames = data_train[idx*batch_size : (idx+1)*batch_size]
            #batch = [imread(path) for path in batch_filenames]
            batch = [Image.open(path) for path in batch_filenames]
            batch_np = [np.array(im_obj) for im_obj in batch]
            batch_images = np.array(batch_np).astype(np.float32)/127.5-1
            
            # downsample images before feeding to generator
            # ToDo: should do this before training
            #batch_inputs = [imresize(im, 0.25) for im in batch]
            batch_inputs = [np.array(im.resize((image_h/4, image_w/4), Image.BICUBIC)) for im in batch]
            batch_inputs = np.array(batch_inputs).astype(np.float32)/127.5-1

            rand_idx = np.random.randint(len(data_test)-batch_size-1)
            disc_batch_files = data_test[rand_idx: rand_idx+batch_size]
            disc_batch = [imread(path) for path in disc_batch_files]
            disc_batch_images = np.array(disc_batch).astype(np.float32)/127.5-1

            #pdb.set_trace()
            sess.run([d_optim], feed_dict={ inputs: batch_inputs, real_ims: disc_batch_images})


            # Update G network
            sess.run([g_optim], feed_dict={ inputs: batch_inputs, real_ims: disc_batch_images})
            # Run g_optim twice to make sure that d_loss does not go to zero
            sess.run([g_optim], feed_dict={ inputs: batch_inputs, real_ims: disc_batch_images})

            errD_fake = d_loss_fake.eval({inputs: batch_inputs})
            errD_real = d_loss_real.eval({real_ims: disc_batch_images})


            errG = g_loss.eval({ inputs: batch_inputs, real_ims: disc_batch_images})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, idx, num_batches,
                    time.time() - start_time, errD_fake+errD_real, errG))

            # every 500 steps, save some images to see performance of network
            if np.mod(counter, 500) == 1:

                rand_idx = np.random.randint(len(data_train))
                sample_image_orig = Image.open(data_train[rand_idx])
                # looks nasty, but we are making it 4D
                sample_image = np.array([np.array(sample_image_orig.resize((image_h/4, image_w/4), Image.BICUBIC))]).astype(np.float32)
                sample_input = sample_image/127.5-1
                real_input = (np.array([np.array(sample_image_orig)]))/127.5-1
                #sample = sess.run([gen_test], feed_dict={inputs: (sample_image/127.5-1)})
                sample, loss = sess.run([gen_test, g_loss], feed_dict={inputs: sample_input, real_ims: real_input})
                print "Sample loss: ", loss
                sample = [sample]

                # save an image, with the original next to the generated one
                resz_input = sample_image[0].repeat(axis=0,repeats=4).repeat(axis=1,repeats=4)
                merge_im = np.zeros( (image_h, image_h*3, 3) )
                merge_im[:, :image_h, :] = np.array(sample_image_orig)
                merge_im[:, image_h:image_h*2, :] = resz_input
                merge_im[:, image_h*2:, :] = (sample[0][0]+1)*127.5
                imsave('./samples/train_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)


            if np.mod(counter, 1000) == 2:
                weight_saver.save(sess, 'checkpoint_bn/model', counter)
                print "saving a checkpoint"

