

from scipy.misc import imread, imshow, imresize
import os
from glob import glob
import numpy as np
# this for sklearn 0.17, for 0.18: use sklearn.model_selection
from sklearn.cross_validation import train_test_split
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim

def print_shape(t):
    print(t.name, t.get_shape().as_list())



########
# CONFIG 
#########
adam_learning_rate = 0.0002
adam_beta1 = 0.5

batch_size = 32
image_h = 128
image_w = 128

num_epochs = 20

###############################
# BUILDING THE MODEL
###############################

#tf.reset_default_graph()

real_ims = tf.placeholder(tf.float32, [batch_size, image_h, image_w, 3], name='real_ims')

# the input to the generator is a downsampled version of the real image
inputs = tf.placeholder(tf.float32, [batch_size, image_h/4, image_w/4, 3], name='inputs')

# add biases???

# generator section
print "GENERATOR"
print "-----------"
with tf.variable_scope("generator") as scope:
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
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
        net = inputs
        print_shape(net)

        # encoder section
        net = slim.conv2d(net, 256, [3, 3], scope='conv1')
        print_shape(net)

        net = slim.conv2d(net, 1024, [3, 3], scope='conv2')
        print_shape(net)

        # decoder section
        net = slim.conv2d_transpose(net, 512, [5, 5], scope='deconv3')
        print_shape(net)

        net = slim.conv2d_transpose(net, 256, [5, 5], scope='deconv4')
        print_shape(net)

        net = slim.conv2d_transpose(net, 128, [5, 5], scope='deconv5')
        print_shape(net)

        # tanh since images have range [-1,1]
        net = slim.conv2d_transpose(net, 3, [5, 5], scope='deconv6', activation_fn=tf.nn.tanh)
        print_shape(net)

        # generate downsampled outputs
        outputs = tf.image.resize_images(net, [image_h/4, image_w/4])
        print_shape(outputs)
    
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

    disc_logits = slim.fully_connected(tf.reshape(disc, [-1, 8192]), 1, activation_fn=None, scope='fc6')
    return disc_logits

# create 2 discriminators: for fake and real images
with tf.variable_scope("discriminators") as scope:
    disc_real = create_discriminator(real_ims)
    scope.reuse_variables()
    disc_fake = create_discriminator(net)
    


# Losses
###########

# loss on real input images; all outputs should be 1
d_loss_real = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_real, \
                                         tf.ones_like(disc_real)))

# loss on fake input images, fakes should be 0
d_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, \
                                            tf.zeros_like(disc_fake)))
# similar to above, but we want fake (generator) images to output 1
g_loss = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, \
                                            tf.ones_like(disc_fake)))

# ToDo: add L2 loss for generator between input & output image

# d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
# d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

d_loss = d_loss_real + d_loss_fake

# g_loss_sum = tf.summary.scalar("g_loss", g_loss)
# d_loss_sum = tf.summary.scalar("d_loss", d_loss)


train_vars = tf.trainable_variables()

d_vars = [var for var in train_vars if 'discriminators' in var.name]
g_vars = [var for var in train_vars if 'generator' in var.name]

#print d_vars
#print g_vars

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

print "TRAINING"
print "-----------"

start_time = time.time()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    for epoch in range(num_epochs):

        for idx in xrange(num_batches):
            batch_filenames = data_train[idx*batch_size : (idx+1)*batch_size]
            batch = [imread(path).astype(np.float) for path in batch_filenames]
            batch_images = np.array(batch).astype(np.float32)
            
            # downsample images before feeding to generator
            # ToDo: should do this before training
            batch_inputs = [imresize(im, 0.25) for im in batch]
            batch_inputs = np.array(batch_inputs).astype(np.float32)
            

            # Update D network
            sess.run([d_optim],
                feed_dict={ real_ims: batch_images, inputs: batch_inputs })
            #self.writer.add_summary(summary_str, counter)

            # Update G network
            sess.run([g_optim],
                feed_dict={ inputs: batch_inputs })
            #self.writer.add_summary(summary_str, counter)

            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            sess.run([g_optim],
                feed_dict={ inputs: batch_inputs })
            #self.writer.add_summary(summary_str, counter)

            errD_fake = d_loss_fake.eval({inputs: batch_inputs})
            errD_real = d_loss_real.eval({real_ims: batch_images})
            errG = g_loss.eval({inputs: batch_inputs})

            #counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, idx, num_batches,
                    time.time() - start_time, errD_fake+errD_real, errG))

#             # every 100 steps, save some images to see performance of network
#             if np.mod(counter, 100) == 1:
#                 samples, d_loss, g_loss = self.sess.run(
#                     [self.sampler, self.d_loss, self.g_loss],
#                     feed_dict={self.z: sample_z, self.images: sample_images}
#                 )
#                 save_images(samples, [8, 8],
#                             './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
#                 print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

#             if np.mod(counter, 500) == 2:
#                 self.save(config.checkpoint_dir, counter)