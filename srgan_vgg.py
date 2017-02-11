

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

def create_VGG5_4(inputs):
    with slim.arg_scope([slim.conv2d], trainable=False):
        net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_4')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_4')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_4')

    return net

# need to convert images to 224x224 before input to VGGnet
def vgg_process_image(img):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    '''

    scale = 256
    crop = 224

    # isotropic Rescale
    img_shape = tf.to_float(tf.shape(img)[:2])
    min_length = tf.minimum(img_shape[0], img_shape[1])
    new_shape = tf.to_int32((scale / min_length) * img_shape)

    img = tf.image.resize_images(img, [new_shape[0], new_shape[1]])
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    # See: https://github.com/tensorflow/tensorflow/issues/521
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.pack([0, offset[0], offset[1], 0]), size=tf.pack([-1, crop, crop, -1]))

    # # Mean subtraction
    # return tf.to_float(img) - mean
    return tf.to_float(img)


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


# ToDo: activation before batch norm?
def create_discriminator(inputs):
    with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    activation_fn=lrelu,
                    normalizer_fn=slim.batch_norm,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    weights_regularizer=slim.l2_regularizer(0.0005)):

        disc = slim.conv2d(inputs, 64, [3, 3], normalizer_fn=None, scope='conv1')
        print_shape(disc)

        disc = slim.conv2d(disc, 64, [3, 3], stride=2, scope='conv2')
        print_shape(disc)

        disc = slim.conv2d(disc, 128, [3, 3], scope='conv3')
        print_shape(disc)

        disc = slim.conv2d(disc, 128, [3, 3], stride=2, scope='conv4')
        print_shape(disc)

        disc = slim.conv2d(disc, 256, [3, 3], scope='conv5')
        print_shape(disc)

        disc = slim.conv2d(disc, 256, [3, 3], stride=2, scope='conv6')
        print_shape(disc)

        disc = slim.conv2d(disc, 512, [3, 3], scope='conv7')
        print_shape(disc)

        disc = slim.conv2d(disc, 512, [3, 3], stride=2, scope='conv8')
        print_shape(disc)

        disc = slim.fully_connected(tf.contrib.layers.flatten(disc), 1024, normalizer_fn=None, scope='fc9')
        print_shape(disc)

    disc_logits = slim.fully_connected(tf.contrib.layers.flatten(disc), 1, activation_fn=None, scope='fc10')
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

# loss between output feature maps of VGGnet
with tf.variable_scope("") as scope:
    vgg_real = create_VGG5_4(vgg_process_image(real_ims))
    scope.reuse_variables()
    vgg_fake = create_VGG5_4(vgg_process_image(gen))

diff_mse = tf.contrib.layers.flatten(vgg_real - vgg_fake)
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

b_load = True

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    for epoch in range(num_epochs):

        np.random.shuffle(data_train)

        # load VGG weights
        optimistic_restore(sess, '/home/wseto/dcgan/vgg_checkpoint/model-1')
        print "loaded vgg weights"

        if b_load:
            ckpt = tf.train.get_checkpoint_state('/home/wseto/dcgan/checkpoint')
            # weight_saver.restore(sess, ckpt.model_checkpoint_path)
            counter = int(ckpt.model_checkpoint_path.split('-', 1)[1])

            optimistic_restore(sess, '/home/wseto/dcgan/checkpoint/model-103002')
            print "successfuly restored!" + " counter:", counter

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

            errD_fake = d_loss_fake.eval({inputs: batch_inputs})
            errD_real = d_loss_real.eval({real_ims: batch_images})
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
                imsave('./samples_gan/train_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)

                #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

            if np.mod(counter, 1000) == 2:
                weight_saver.save(sess, 'checkpoint_gan/model', counter)
                print "saving a checkpoint"

