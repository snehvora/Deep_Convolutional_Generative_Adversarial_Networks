import os
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

IMG_H = 64
IMG_W = 64
IMG_C = 3

def build_discriminator(img_size):
    i=Input(shape=img_size)
    x=Flatten()(i)
    x=Dense(1024,activation=LeakyReLU(alpha=0.2))(x)
    x=Dense(512,activation=LeakyReLU(alpha=0.2))(x)
    x=Dense(256,activation=LeakyReLU(alpha=0.2))(x)
    x=Dense(1,activation='sigmoid')(x)
    model=Model(i,x)
    return model

def build_generator(laten_dim):
    i = Input(shape=(laten_dim,))
    # Reshape input into 7x7x256 tensor via a fully connected layer
    x = Dense(8*8*256,use_bias=False)(i)
    x = Reshape((8,8,256))(x)
    # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
    x = Conv2DTranspose(128, kernel_size = 3, strides = 2, padding='same')(x)
    #Batch normalization
    x = BatchNormalization()(x)
    #Leaky ReLU
    x = LeakyReLU(alpha=0.01)(x)
    # Transposed convolution layer, from 14x14x128 to 14x14x64 tensor
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    # Batch normalization
    x = BatchNormalization()(x)
    # Leaky ReLU
    x = LeakyReLU(alpha=0.01)(x)
    # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
    x = Conv2DTranspose(3, kernel_size = 3, strides = 2, padding='same')(x)
    # Tanh activation
    x = Activation('tanh')(x)
    model = Model(i,x)
    
    return model


def sample_images(epoch):
    rows,cols=5,5
    noise=np.random.randn(rows*cols,laten_dim)
    fake_img=generator.predict(noise)

    fake_img=(0.5*fake_img)+0.5
    figs,axs=plt.subplots(5,5)
    idx=0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(fake_img[idx].reshape(IMG_H,IMG_W,IMG_C))
            axs[i,j].axis('off')
            idx =idx + 1
    figs.savefig("gan_images/{}.png".format(epoch))
    plt.close()

def tf_dataset(image_path, batch_size):
    dataset = []
    for i in image_path:
        d = cv2.imread(i)
        d = cv2.resize(d,(64,64))
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        dataset.append(d)
    dataset = np.array(dataset).reshape(len(dataset),64,64,3)
    dataset = shuffle(dataset)
    dataset = dataset/255
    return dataset

if __name__ =="__main__":
    # hyperparameters
    batch_size = 64
    laten_dim = 128
    num_epochs = 10000000
    images_path = glob("data/*")
    x_train = tf_dataset(images_path, batch_size)
    
    img_size = x_train[0].shape
    descriminator = build_discriminator(img_size)
    descriminator.compile(optimizer = 'adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    generator = build_generator(laten_dim)
    z=Input(shape=(laten_dim,))
    img = generator(z)
    descriminator.trainable = False
    fake_prediction = descriminator(img)
    combined_model = Model(z,fake_prediction)
    combined_model.compile(optimizer = 'adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    if not os.path.exists('gan_images'):
        os.makedirs('gan_images')

    # training
    ones = np.ones((batch_size,))
    zeros = np.zeros((batch_size,))                    
    d_losses=[]
    g_losses=[]
    
    for epoch in range(num_epochs):
        #1st> Train the discriminator
        idx = np.random.randint(0,x_train.shape[0],batch_size)
        real_img = x_train[idx]

        noise=np.random.randn(batch_size,laten_dim)
        fake_img=generator.predict(noise)

        d_loss_real,d_acc_real=descriminator.train_on_batch(real_img,ones)
        d_loss_fake,d_acc_fake=descriminator.train_on_batch(fake_img,zeros)
        d_loss=0.5*(d_loss_real+d_loss_fake)
        d_acc=0.5*(d_acc_real+d_acc_fake)

        #2nd train generator
        noise=np.random.randn(batch_size,laten_dim)
        g_loss=combined_model.train_on_batch(noise,ones)

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if epoch%500==0:
            sample_images(epoch)

        if epoch%10000==0:
            generator.save('genrator_model')

        if epoch%2==0:
            epoch=epoch+1
            print("epoch:{}/{}, d_loss:{}, d_acc:{}, g_loss:{}".format(epoch,num_epochs,d_loss,d_acc,g_loss))

