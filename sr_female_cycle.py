# experiment: trying cyclegan for male -> female

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

def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])    
    restore_vars = []    
    for var_name, saved_var_name in var_names:            
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

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

inputs = tf.placeholder(tf.float32, [None, image_h, image_w, 3], name='inputs')


# generator section
print "GENERATOR"
print "-----------"

nb_residual = 5
def create_generator(inputs, b_training=True):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with slim.arg_scope([slim.batch_norm],
                    is_training=b_training, scale=True, decay=0.9):
            net = inputs
            print_shape(net)

            # need to add some conv layers here since input is not downsampled anymore
            net = tf.nn.relu(slim.conv2d(net, 16, [3, 3], scope='pre_gconv1'))
            print_shape(net)

            net = tf.nn.relu(slim.conv2d(net, 32, [3, 3], stride=2, scope='pre_gconv2'))
            print_shape(net)

            net = tf.nn.relu(slim.conv2d(net, 64, [3, 3], stride=2, scope='gconv1'))
            print_shape(net)

            net1 = net

            res_inputs = net1
            for n in range(nb_residual):
                net = tf.nn.relu(slim.batch_norm(slim.conv2d(res_inputs, 64, [3, 3], scope='conv1_res'+str(n)), scope='bn_'+str(n*2)))
                net = slim.batch_norm(slim.conv2d(net, 64, [3, 3], scope='conv2_res'+str(n)), scope='bn_'+str(n*2+1))
                net = net + res_inputs
                res_inputs = net


            print_shape(net)

            net = slim.batch_norm(slim.conv2d(net, 64, [3, 3], scope='gconv2'), scope='bn_'+str(nb_residual*2)) + net1
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


# create cycle generator
with tf.variable_scope("cycle_generator") as scope:
    gen_cycle = create_generator(gen)



print "DISCRIMINATOR"
print "--------------"

# anneal our noise stddev
decay_counter = tf.Variable(0, name="counter", dtype=tf.float32)

def create_discriminator(inputs, counter, b_reuse=False):
    with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    stride=2):

        # 0.05 starting stddev
        #noise_stddev = tf.constant(0.05, dtype=tf.float32) - tf.constant(5.5e-7, dtype=tf.float32)*counter
        #noisy_inputs = inputs + tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=noise_stddev, dtype=tf.float32)
        noisy_inputs = inputs

        disc = lrelu(slim.conv2d(noisy_inputs, 64, [5, 5], scope='conv1'))
        print_shape(disc)

        with slim.arg_scope([slim.batch_norm], scale=True, decay=0.9):
            disc = lrelu(slim.batch_norm(slim.conv2d(disc, 128, [5, 5], scope='conv2'), scope='d_bn1'))
            print_shape(disc)

            disc = lrelu(slim.batch_norm(slim.conv2d(disc, 256, [5, 5], scope='conv3'), scope='d_bn2'))
            print_shape(disc)

            disc = lrelu(slim.batch_norm(slim.conv2d(disc, 512, [5, 5], scope='conv4'), scope='d_bn3'))
            print_shape(disc)

            disc = lrelu(slim.batch_norm(slim.conv2d(disc, 512, [5, 5], scope='conv5'), scope='d_bn4'))
            print_shape(disc)

    disc = lrelu(slim.fully_connected(tf.contrib.layers.flatten(disc), 1024, activation_fn=None, scope='fc6'))
    disc_logits = slim.fully_connected(tf.contrib.layers.flatten(disc), 1, activation_fn=None, scope='fc7')

    return disc_logits
# create 2 discriminators: for fake and real images
with tf.variable_scope("discriminators") as scope:
    disc_real = create_discriminator(real_ims, decay_counter)
    scope.reuse_variables()
    disc_fake = create_discriminator(gen, decay_counter, True)


# Losses
###########

# loss on real input images; all outputs should be 1

# make labels noisy for discriminator
rand_val = tf.random_uniform([], seed=42)
labels_real = tf.cond(rand_val < 1, lambda: tf.ones_like(disc_real), lambda: tf.zeros_like(disc_real))
labels_fake = tf.cond(rand_val < 1, lambda: tf.zeros_like(disc_fake), lambda: tf.ones_like(disc_fake))

d_loss_real = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=labels_real))

# loss on fake input images, fakes should be 0
d_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=labels_fake))

# similar to above, but we want fake (generator) images to output 1
g_loss_adv = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))



# reconstruction loss is on input and cycle output
gen_mse_HR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen_cycle - inputs)), 1)
# average for the batch
gen_L2_HR = tf.reduce_mean(gen_mse_HR)


train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminators' in var.name]
g_vars = [var for var in train_vars if 'generator' in var.name]


# optimize the generator and discriminator separately
g_loss = gen_L2_HR + 0.01*g_loss_adv
d_loss = d_loss_real + d_loss_fake

optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1)
d_train_op = slim.learning.create_train_op(d_loss, optim, variables_to_train=d_vars)
g_train_op = slim.learning.create_train_op(g_loss, optim, variables_to_train=g_vars)

    
weight_saver = tf.train.Saver(max_to_keep=1)


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
samples_dir = 'samples_female_cycle'
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

checkpoint_dir = 'checkpoint_female_cycle'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


print "TRAINING"
print "-----------"

start_time = time.time()
counter = 0

b_load = False
#ckpt_dir = '/home/wseto/dcgan/checkpoint_up_rand'

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    if b_load:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        weight_saver.restore(sess, ckpt.model_checkpoint_path)
        counter = int(ckpt.model_checkpoint_path.split('-', 1)[1]) 
        print "successfully restored!" + " counter:", counter
        
    for epoch in range(num_epochs):

        np.random.shuffle(data_train)

        total_errD = 2

        for idx in xrange(num_batches):
            batch_filenames = data_train[idx*batch_size : (idx+1)*batch_size]
            
            batch_origs, _ = get_images(batch_filenames)
            
            # discriminator batch is different since we are doing unpaired experiment
            rand_idx = np.random.randint(len(data_disc)-batch_size-1)
            disc_batch_files = data_disc[rand_idx: rand_idx+batch_size]     
            disc_batch_orig, _ = get_images(disc_batch_files)

            fetches = [d_loss_fake, d_loss_real, g_loss_adv, d_train_op, g_train_op]
            errD_fake, errD_real, errG, _, _ = sess.run(fetches, feed_dict={ inputs: batch_origs, real_ims: disc_batch_orig})

            # if total_errD > 1:
            #     fetches = [d_loss_fake, d_loss_real, g_loss_adv, d_optim, g_optim]
            #     errD_fake, errD_real, errG, _, _ = sess.run(fetches, feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig})
            # else:
            #     fetches = [d_loss_fake, d_loss_real, g_loss_adv, g_optim]
            #     errD_fake, errD_real, errG, _ = sess.run(fetches, feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig})

            total_errD = errD_fake + errD_real



            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_fake: %.3f, d_loss_real: %.3f, g_loss: %.8f" \
                % (epoch, idx, num_batches,
                    time.time() - start_time, errD_fake, errD_real, errG))


            if np.mod(counter, 30) == 1:


                rand_idx = np.random.randint(len(data_sample)-batch_size+1)
                sample_origs, _ = get_images(data_sample[rand_idx: rand_idx+batch_size])

                sample = sess.run([gen_test], feed_dict={inputs: sample_origs})

                sample_cycle = sess.run([gen_cycle], feed_dict={inputs: sample_origs})
                
                # save an image, with the original next to the generated one
                merge_im = np.zeros( (image_h, image_h*3, 3) )
                merge_im[:, :image_h, :] = (sample_origs[0]+1)*127.5
                merge_im[:, image_h:image_h*2, :] = (sample[0][0]+1)*127.5
                merge_im[:, image_h*2:, :] = (sample_cycle[0][0]+1)*127.5

                imsave(samples_dir + '/test_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)
                print "saving a sample"

            if np.mod(counter, 1000) == 2:
                weight_saver.save(sess, checkpoint_dir + '/model', counter)
                print "saving a checkpoint"