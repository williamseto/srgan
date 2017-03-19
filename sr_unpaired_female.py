
# experiment: unpaired GAN with L2 loss on the low resolution images and
# only female images in the discriminator

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

# takes list of filenames and returns a 4D batch of images
# [N x W x H x C]
# also resize if necessary
def get_images(filenames, imsize=None):

    if imsize:
        batch_orig = [imresize(imread(path), (imsize, imsize), interp='bicubic') for path in filenames]
    else:
        batch_orig = [imread(path)for path in filenames]

    batch_orig_normed = np.array(batch_orig).astype(np.float32)/127.5-1

    batch_inputs = [imresize(im, 0.25, interp='bicubic') for im in batch_orig]
    # imresize returns in [0-255] so we have to normalize again
    batch_inputs_normed = np.array(batch_inputs).astype(np.float32)/127.5-1

    return batch_orig_normed, batch_inputs_normed

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


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

# this is really bad
batch_norm_list = []
nb_residual = 8
n_extra_bn = 1
for n in range(nb_residual*2 + n_extra_bn):
    batch_norm_list.append(batch_norm(name='bn'+str(n)))

def create_generator(inputs, b_training=True):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        normalizer_params={'scale':True, 'decay': 0.9, 'updates_collections': None},
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = inputs
        print_shape(net)

        net = tf.nn.relu(slim.conv2d(net, 64, [3, 3], scope='gconv1'))
        print_shape(net)

        net1 = net

        res_inputs = net1
        for n in range(nb_residual):
            net = tf.nn.relu(batch_norm_list[n*2](slim.conv2d(res_inputs, 64, [3, 3], scope='conv1_res'+str(n)), train=b_training))
            net = batch_norm_list[n*2+1](slim.conv2d(net, 64, [3, 3], scope='conv2_res'+str(n)), train=b_training)
            net = net + res_inputs
            res_inputs = net


        print_shape(net)

        net = batch_norm_list[-1](slim.conv2d(net, 64, [3, 3], scope='gconv2'), train=b_training) + net1
        print_shape(net)

        # deconv
        net = tf.nn.relu(slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv1'))
        print_shape(net)

        net = tf.nn.relu(slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv2'))
        print_shape(net)


        # tanh since images have range [-1,1]
        net = slim.conv2d(net, 3, [3, 3], scope='gconv3', activation_fn=tf.nn.tanh)
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

# make labels noisy for discriminator
rand_val = tf.random_uniform([], seed=42)
labels_real = tf.cond(rand_val < 0.95, lambda: tf.ones_like(disc_real), lambda: tf.zeros_like(disc_real))
labels_fake = tf.cond(rand_val < 0.95, lambda: tf.zeros_like(disc_fake), lambda: tf.ones_like(disc_fake))

d_loss_real = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=labels_real))

# loss on fake input images, fakes should be 0
d_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=labels_fake))

# similar to above, but we want fake (generator) images to output 1
g_loss_adv = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))


# Losses

# metric: L2 between downsampled generated output and input
gen_LR = tf.image.resize_images(gen, [image_h/4, image_w/4])
gen_mse_LR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen_LR - inputs)), 1)
gen_L2_LR = tf.reduce_mean(gen_mse_LR)

# metric: L2 between generated output and the original image
gen_mse_HR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen - real_ims)), 1)
# average for the batch
gen_L2_HR = tf.reduce_mean(gen_mse_HR)

# metric: PSNR between generated output and original input
gen_rmse_HR = tf.sqrt(gen_mse_HR)
gen_PSNR = tf.reduce_mean(20*tf.log(1.0/gen_rmse_HR)/tf.log(tf.constant(10, dtype=tf.float32)))

# baselines: L2 and PSNR between bicubic upsampled input and original image
upsampled_output = tf.image.resize_bicubic(inputs, [image_h, image_h])
ups_mse_HR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(upsampled_output - real_ims)), 1)
ups_L2_HR = tf.reduce_mean(ups_mse_HR)

ups_rmse_HR = tf.sqrt(ups_mse_HR)
ups_PSNR = tf.reduce_mean(20*tf.log(1.0/ups_rmse_HR)/tf.log(tf.constant(10, dtype=tf.float32)))

# ToDo: add SSIM metric



train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminators' in var.name]
g_vars = [var for var in train_vars if 'generator' in var.name]


# optimize the generator and discriminator separately
g_loss = gen_L2_LR + 0.05*g_loss_adv
d_loss = d_loss_real + d_loss_fake

d_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(g_loss, var_list=g_vars)
    
weight_saver = tf.train.Saver(max_to_keep=1)


# logging

tf.summary.scalar("d_loss", d_loss)
tf.summary.scalar("g_loss_L2_LR", gen_L2_LR)
tf.summary.scalar("g_loss_adv", g_loss_adv)

tf.summary.scalar("gen_L2_HR", gen_L2_HR)
tf.summary.scalar("gen_PSNR_HR", gen_PSNR)
tf.summary.scalar("ups_L2_HR", ups_L2_HR)
tf.summary.scalar("ups_PSNR_HR", ups_PSNR)

merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs_f_128/train')
test_writer = tf.summary.FileWriter('./logs_f_128/test')



print "initialization done"


#############
# TRAINING
############

female_data_dir = '/home/wseto/datasets/celeba_female'
male_data_dir = '/home/wseto/datasets/celeba_male'

female_data = glob(os.path.join(female_data_dir, "*.png"))
male_data = glob(os.path.join(male_data_dir, "*.png"))

data_disc, female_data_nondisc = train_test_split(female_data, test_size=0.5, random_state=42)

female_data_train, female_data_sample = train_test_split(female_data_nondisc, test_size=0.1, random_state=42)
male_data_train, male_data_sample = train_test_split(male_data, test_size=0.1, random_state=42)

data_train = female_data_train + male_data_train
data_sample = female_data_sample + male_data_sample

print "data train:", len(data_train)
print "data disc:", len(data_disc)
print "data sample:", len(data_sample)

# create directories to save checkpoint and samples
samples_dir = 'samples_female'
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

checkpoint_dir = 'checkpoint_female'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print "TRAINING"
print "-----------"

start_time = time.time()
counter = 0

b_load = False
#ckpt_dir = '/home/wseto/dcgan/checkpoint_bn'


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    for epoch in range(num_epochs):

        np.random.shuffle(data_train)

        if b_load:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            weight_saver.restore(sess, ckpt.model_checkpoint_path)
            counter = int(ckpt.model_checkpoint_path.split('-', 1)[1]) 
            print "successfuly restored!" + " counter:", counter

        for idx in xrange(num_batches):
            batch_filenames = data_train[idx*batch_size : (idx+1)*batch_size]
            
            batch_origs, batch_inputs = get_images(batch_filenames)
            
            # discriminator batch is different since we are doing unpaired experiment
            rand_idx = np.random.randint(len(data_disc)-batch_size-1)
            disc_batch_files = data_disc[rand_idx: rand_idx+batch_size]     
            disc_batch_orig, disc_batch_inputs = get_images(disc_batch_files, imsize=image_h)

            # Update D network
            sess.run([d_optim], feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig})
            # Update G network
            sess.run([g_optim], feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig})
            # Run g_optim twice to make sure that d_loss does not go to zero
            sess.run([g_optim], feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig})


            errD_fake = d_loss_fake.eval({inputs: batch_inputs})
            errD_real = d_loss_real.eval({real_ims: disc_batch_orig})
            errG = g_loss.eval({ inputs: batch_inputs, real_ims: disc_batch_orig})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, idx, num_batches,
                    time.time() - start_time, errD_fake+errD_real, errG))



            if np.mod(counter, 30) == 1:

                # only for the purposes of computing metrics, pass in original images
                train_summary = sess.run([merged_summary], feed_dict={ inputs: batch_inputs, real_ims: batch_origs})
                train_writer.add_summary(train_summary[0], counter)

                rand_idx = np.random.randint(len(data_sample)-batch_size+1)
                sample_origs, sample_inputs = get_images(data_sample[rand_idx: rand_idx+batch_size])

                sample, loss = sess.run([gen_test, g_loss], feed_dict={inputs: sample_inputs, real_ims: disc_batch_orig})
                print "Sample loss: ", loss

                img_summary = tf.summary.image("test_image{:06d}".format(counter), gen_test)
                test_summary, test_img_summary = sess.run([merged_summary, img_summary], feed_dict={ inputs: sample_inputs, real_ims: sample_origs})
                test_writer.add_summary(test_summary, counter)


                sample = [sample]
                # save an image, with the original next to the generated one
                resz_input = sample_inputs[0].repeat(axis=0,repeats=4).repeat(axis=1,repeats=4)
                merge_im = np.zeros( (image_h, image_h*3, 3) )
                merge_im[:, :image_h, :] = (sample_origs[0]+1)*127.5
                merge_im[:, image_h:image_h*2, :] = (resz_input+1)*127.5
                merge_im[:, image_h*2:, :] = (sample[0][0]+1)*127.5
                imsave(samples_dir + '/test_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)

            if np.mod(counter, 1000) == 2:
                weight_saver.save(sess, checkpoint_dir + '/model', counter)
                print "saving a checkpoint"


