from keras.layers import Conv2D, BatchNormalization, Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU


#DISCRIMINATOR
def build_discriminator (start_filters, spatial_dim, filter_size):

    def add_discriminator_block(x, filters, filter_size):
        x = Conv2D(filters, filter_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        return x


    inp = Input(shape=spatial_dim, spatial_dim, 3))

    x = add_discriminator_block(inp, start_filters, filter_size)
    x = add_discriminator_block(x, start_filters * 2, filter_size)
    x = add_discriminator_block(x, start_filters * 4, filter_size)
    x = add_discriminator_block(x, start_filters * 8, filter_size)


    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation= 'sigmoid')(x)

    return Model(inputs=inp, outputs=x)


#GENERATOR
from keras.layers import Deconvolution2D, Reshape

def build_generator(start_filters, filter_size, latent_dim):

    def add_generator_block(x, filters, filter_size):
        x = Deconvolution2D(filters, filter_size, filter_size, latent_dim)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        return x

    inp = Input(shape=(latent_dim,))

    x = Dense(4 * 4 * (start_filters * 8), input_dim=latent_dim)(inp)
    x = BatchNormalization()(x)
    x = Reshape(target_shape=(4, 4, start_filters * 8))(x)

    x = add_generator_block(x, start_filters * 4, filter_size)
    x = add_generator_block(x, start_filters * 2, filter_size)
    x = add_generator_block(x, start_filters, filter_size)
    x = add_generator_block(x, start_filters, filter_size)

    x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)

    return Model(inputs=inp, outputs=x)

#GAN
import pandas as pd
import os
from keras.optimizers import Adam

df_celeb = pd.read_csv('list_attr_celeba.csv')
TOTAL_SAMPLES = df_celeb.shape[0]
SPATIAL_DIM = 64
LATENT_DIM_GAN = 100
FILTER_SIZE = 5
NET_CAPACITY = 16
BATCH_SIZE_GAN = 32
PROGRESS_INTERVAL = 80
ROOT_DIR = "visualization"
if not os.path.isdir(ROOT.DIR):
    os.mkdr(ROOT_DIR)



def construct_models(verbose=False):
    discriminator = build_discriminator(NET_CAPACITY,)
