

from scipy.misc import imread, imshow, imresize, imsave
import os
from glob import glob
import numpy as np
# this for sklearn 0.17, for 0.18: use sklearn.model_selection
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim

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

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def residual(inputs, name="res"):
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params={'epsilon':1e-5, 'scale':True}):

        net = slim.conv2d(inputs, 64, [3, 3])
        net = slim.conv2d(net, 64, [3, 3], activation_fn=None)
        net = net + inputs
        return net



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


real_ims = tf.placeholder(tf.float32, [batch_size, image_h, image_w, 3], name='real_ims')

# the input to the generator is a downsampled version of the real image
inputs = tf.placeholder(tf.float32, [None, image_h/4, image_w/4, 3], name='inputs')


# generator section
print "GENERATOR"
print "-----------"

def create_generator(inputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = inputs
        print_shape(net)

        net = slim.conv2d(net, 64, [3, 3], scope='conv1')
        print_shape(net)

        net1 = net

        nb_residual = 10
        for n in range(nb_residual):
            net = residual(net1, "res" + str(n))

        print_shape(net)

        net = slim.conv2d(net, 64, [3, 3], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv2') + net1
        print_shape(net)

        # deconv
        net = slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv1')
        print_shape(net)

        net = slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv2')
        print_shape(net)


        # tanh since images have range [-1,1]
        net = slim.conv2d(net, 3, [3, 3], scope='conv3', activation_fn=tf.nn.tanh)
        print_shape(net)

        return net

with tf.variable_scope("generator") as scope:
    gen = create_generator(inputs)
    
print "DISCRIMINATOR"
print "--------------"

def create_discriminator(inputs):
    with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training':True,
                                       'decay':0.997,
                                       'epsilon':1e-5,
                                       'scale':True},
                    stride=2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.0001),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
        disc = slim.conv2d(inputs, 64, [5, 5], scope='conv1')
        print_shape(disc)

        disc = slim.conv2d(disc, 128, [5, 5], scope='conv2')
        print_shape(disc)

        disc = slim.conv2d(disc, 256, [5, 5], scope='conv3')
        print_shape(disc)

        disc = slim.conv2d(disc, 512, [5, 5], scope='conv4')
        print_shape(disc)

        disc = slim.conv2d(disc, 512, [5, 5], scope='conv5')
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
d_loss_real = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))

# loss on fake input images, fakes should be 0
d_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))

# similar to above, but we want fake (generator) images to output 1
g_loss_adv = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))

# L2 LOSS

diff_mse = tf.contrib.layers.flatten(gen - real_ims)
sum_mse = tf.reduce_sum(tf.square(diff_mse), 1)
g_loss_L2 = tf.reduce_mean(sum_mse)


g_loss = g_loss_L2
# d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
# d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

d_loss = d_loss_real + d_loss_fake

# g_loss_sum = tf.summary.scalar("g_loss", g_loss)
# d_loss_sum = tf.summary.scalar("d_loss", d_loss)


train_vars = tf.trainable_variables()

d_vars = [var for var in train_vars if 'discriminators' in var.name]
g_vars = [var for var in train_vars if 'generator' in var.name]


# optimizer the generator and discriminator separately
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
data = glob(os.path.join(data_dir, "*.png"))
data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)


print "TRAINING"
print "-----------"

start_time = time.time()
counter = 0

b_load = False

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    for epoch in range(num_epochs):

        np.random.shuffle(data_train)

        for idx in xrange(num_batches):
            batch_filenames = data_train[idx*batch_size : (idx+1)*batch_size]
            batch = [imread(path) for path in batch_filenames]
            batch_images = np.array(batch).astype(np.float32)/127.5-1
            
            # downsample images before feeding to generator
            # ToDo: should do this before training
            batch_inputs = [imresize(im, 0.25) for im in batch]
            batch_inputs = np.array(batch_inputs).astype(np.float32)/127.5-1
            

            # Update D network
            #sess.run([d_optim], feed_dict={ real_ims: batch_images, inputs: batch_inputs })


            # Update G network
            sess.run([g_optim], feed_dict={ inputs: batch_inputs, real_ims: batch_images})

            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            #sess.run([g_optim], feed_dict={ inputs: batch_inputs })

            errD_fake = 0 #d_loss_fake.eval({inputs: batch_inputs})
            errD_real = 0 #d_loss_real.eval({real_ims: batch_images})
            errG = g_loss.eval({ inputs: batch_inputs, real_ims: batch_images})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, idx, num_batches,
                    time.time() - start_time, errD_fake+errD_real, errG))

            # every 500 steps, save some images to see performance of network
            if np.mod(counter, 25) == 1:

                rand_idx = np.random.randint(len(data_test))
                sample_image_orig = imread(data_test[rand_idx]).astype(np.float)
                sample_image = np.array([imresize(sample_image_orig, 0.25)]).astype(np.float32)
                
                sample = sess.run([gen], feed_dict={inputs: (sample_image/127.5-1)})

                # save an image, with the original next to the generated one
                resz_input = sample_image.repeat(axis=1,repeats=4).repeat(axis=2,repeats=4)
                merge_im = np.zeros( (128, 384, 3) )
                merge_im[:, :128, :] = sample_image_orig
                merge_im[:, 128:256, :] = resz_input
                merge_im[:, 256:, :] = (sample[0][0]+1)*127.5
                imsave('./samples/train_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)

                #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

            if np.mod(counter, 1000) == 2:
                weight_saver.save(sess, 'checkpoint/model', counter)
                print "saving a checkpoint"

