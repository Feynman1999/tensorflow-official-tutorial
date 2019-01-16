from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
tf.enable_eager_execution()

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# We are normalizing the images to the range of [-1, 1]
train_images = (train_images - 127.5) / 127.5
noise_dim = 10


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*32, use_bias=False, input_shape=(noise_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Reshape((7, 7, 32)))
    assert model.output_shape == (None, 7, 7, 32) # Note: None is the batch size
    
    # 5 5 256（输入通道数） 128（输出通道数）  
    model.add(tf.keras.layers.Conv2DTranspose(16, (5, 5), strides=(1, 1), padding='same', use_bias=False, data_format='channels_last'))
    # 第三个参数是output_shape 用于检测是否是想要的结果    这里没传参
    assert model.output_shape == (None, 7, 7, 16)  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(6, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 6)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
  
    return model



def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')) # samples*14*14*64
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
      
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')) # samples*7*7*128
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
       
    model.add(tf.keras.layers.Flatten()) # samples * -1
    model.add(tf.keras.layers.Dense(1))
    
    return model



def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(generated_output), logits=generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator) # 优化器 和 模型


BUFFER_SIZE = 60000     
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


EPOCHS = 50
num_examples_to_generate = 16

# We'll re-use this random vector used to seed the generator so
# it will be easier to see the improvement over time.
random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                 noise_dim])  # 16*100



# 对于每个batch 优化网络
def train_step(images):
    # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, noise_dim])
  
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: # as 两次
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True) # 对于每一张图片 （共batch_size个） 计算一个real_output的值
        generated_output = discriminator(generated_images, training=True)
            
        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)
    
    # tf.GradientTape()就被记录下来啦
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))



# 在每个epoch结束时进行测试并保存
def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    generate_images = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))
 
    for i in range(generate_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(generate_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()


# 据说可以起到加速的作用 (⊙o⊙)…
train_step = tf.contrib.eager.defun(train_step)


def train(dataset, epochs):
    num=1
    for epoch in range(epochs):
        start = time.time()

        # 依次喂入每个batch
        for images in dataset:  
            print(num)
            num=num+1
            train_step(images)

        #display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, random_vector_for_generation)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                      time.time()-start))
    # generating after the final epoch
    #display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,    # 下标从1开始
                            random_vector_for_generation)


# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


if __name__ == '__main__':
    
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    train(train_dataset, EPOCHS)

    # 恢复最后一轮的参数
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # 看一下最后一轮
    display_image(EPOCHS)