import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import UpSampling2D, MaxPooling3D,Activation, Add, Multiply, Layer, Conv2D, BatchNormalization, MaxPool2D, Concatenate, Cropping2D, Conv2DTranspose, ConvLSTM2D, Input

#------------------ basic layer --------------------
class CNNLayer(Layer):
    def __init__(self, n_channels, kernel_size=3, n_strides=(1,1), padding='same', activation='relu', use_bias=False, use_batch_norm=True):
        super(CNNLayer, self).__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_strides = n_strides
        self.padding = padding
        self.activation_type = activation
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.conv = Conv2D(n_channels, kernel_size=kernel_size, strides=n_strides, padding=padding, use_bias=use_bias)
        if self.use_batch_norm:
            self.bn = BatchNormalization(axis=-1)
        self.activation = Activation(activation)
    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor, training=training)
        if self.use_batch_norm:
            x = self.bn(x, training=training)
        x = self.activation(x)
        return x
    def get_config(self):
        config = super(CNNLayer, self).get_config()
        config.update({
            'n_channels': self.n_channels,
            'kernel_size': self.kernel_size,
            'n_strides': self.n_strides,
            'padding': self.padding,
            'activation': self.activation_type,
            'use_bias': self.use_bias,
            'use_batch_norm': self.use_batch_norm,
            'conv_config': self.conv.get_config(),
            'bn_config': self.bn.get_config() if self.use_batch_norm else None,
            'activation_config': self.activation.get_config()
        })
        return config
    @classmethod
    def from_config(cls, config):
        conv_config = config.pop('conv_config')
        activation_config = config.pop('activation_config')
        bn_config = config.pop('bn_config', None)
        instance = cls(**config)
        instance.conv = Conv2D.from_config(conv_config)
        instance.activation = Activation.from_config(activation_config)
        instance.bn = BatchNormalization.from_config(bn_config) if bn_config else None
        return instance

class CroppingLayer(Layer):
    def __init__(self, size):
        super(CroppingLayer, self).__init__()
        self.size = size
        self.crop = Cropping2D(cropping=((size, size), (size, size)))
    def call(self, input_tensor, training=False):
        x = self.crop(input_tensor)
        return x
    def get_config(self):
        config = super(CroppingLayer, self).get_config()
        config.update({
            'size': self.size,
            'crop_config': self.crop.get_config(),
        })
        return config
    @classmethod
    def from_config(cls, config):
        crop_config = config.pop('crop_config')
        instance = cls(**config)
        instance.crop = Cropping2D.from_config(crop_config)
        return instance

#------------------ Blocks --------------------
class CNNBlock(Layer):
    def __init__(self, channels, block_type='standard', pool=True):
        super(CNNBlock, self).__init__()
        self.channels = channels
        self.block_type = block_type
        self.pooling = pool
        self.conv_1 = CNNLayer(channels)
        self.conv_2 = CNNLayer(channels)
        if self.block_type == 'residual':
            self.initial_conv = CNNLayer(channels)
        if self.pooling:
            self.max_pool = MaxPool2D(2, 2)
    def call(self, input_tensor, training=False):
        if self.block_type == 'residual':
            x = self.initial_conv(input_tensor, training=training)
            x_1 = self.conv_1(x, training=training)
            x_1 = self.conv_2(x_1, training=training)
            x_add = Add()([x, x_1])
            if self.pooling:
                x_p = self.max_pool(x_add)
                return x_p, x_add
            else:
                return x_add
        elif self.block_type == 'standard':
            x = self.conv_1(input_tensor, training=training)
            x = self.conv_2(x, training=training)
            if self.pooling:
                x_p = self.max_pool(x)
                return x_p, x
            else:
                return x
    def get_config(self):
        config = super(CNNBlock, self).get_config()
        config.update({
            'channels': self.channels,
            'block_type': self.block_type,
            'pooling': self.pooling,
            'conv_1_config': self.conv_1.get_config(),
            'conv_2_config': self.conv_2.get_config(),
            'initial_conv_config': self.initial_conv.get_config() if self.block_type == 'residual' else None,
            'max_pool_config': self.max_pool.get_config() if self.pooling else None,
        })
        return config
    @classmethod
    def from_config(cls, config):
        conv_1_config = config.pop('conv_1_config')
        conv_2_config = config.pop('conv_2_config')
        initial_conv_config = config.pop('initial_conv_config', None)
        max_pool_config = config.pop('max_pool_config', None)
        conv_1 = CNNLayer.from_config(conv_1_config)
        conv_2 = CNNLayer.from_config(conv_2_config)
        initial_conv = CNNLayer.from_config(initial_conv_config) if initial_conv_config else None
        max_pool = MaxPool2D.from_config(max_pool_config) if max_pool_config else None
        instance = cls(**config)
        instance.conv_1 = conv_1
        instance.conv_2 = conv_2
        instance.initial_conv = initial_conv
        instance.max_pool = max_pool
        return instance

class ResidualConvLSTMBlock(tf.keras.layers.Layer):
        def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'):
            super(ResidualConvLSTMBlock, self).__init__()
            self.filters = filters
            self.kernel_size = kernel_size
            self.strides = strides
            self.padding = padding
            self.activation = activation
            self.conv_lstm = ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,padding=self.padding, return_sequences=True)
            self.conv_lstm1 = ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, return_sequences=True)
            self.bn1 = BatchNormalization()
            self.activation1 = Activation(self.activation)
            self.conv_lstm2 = ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,padding=self.padding, return_sequences=True)
            self.bn2 = BatchNormalization()
            self.activation2 = Activation(self.activation)
            self.add = Add()
            self.conv_lstm_skip = ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, return_sequences=False)
            self.max_pool = MaxPooling3D(pool_size=(1, 2, 2))
        def call(self, inputs, training=False):
            # First ConvLSTM block
            x_1 = self.conv_lstm(inputs, training=training)
            x = self.conv_lstm1(x_1, training=training)
            x = self.bn1(x, training=training)
            x = self.activation1(x,training=training)
            # Second ConvLSTM block
            x = self.conv_lstm2(x, training=training)
            x = self.bn2(x, training=training)
            x = self.activation2(x, training=training)
            # Adding the residual
            x_add = self.add([x, x_1])
            x_p = self.max_pool(x_add)
            x_skip = self.conv_lstm_skip(x_add, training=training)
            return x_p, x_skip

class AttentionGate(Layer):
    def __init__(self, channels,size=(2, 2), zoom=True, zoom_size=4):
        super(AttentionGate, self).__init__()
        self.channels = channels
        self.size = size
        self.zoom = zoom
        self.zoom_size = zoom_size
        self.g_path = Conv2D(self.channels, kernel_size=1, strides=(1, 1), padding='same', use_bias=True)
        self.x_path = Conv2D(self.channels, kernel_size=1, strides=(2, 2), padding='same', use_bias=True)
        self.psi = Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=True)
        self.activation_relu = Activation('relu')
        self.activation_sigmoid = Activation('sigmoid')
        self.upsampling = UpSampling2D(size)
        if self.zoom == True:
            self.cropping = CroppingLayer(self.zoom_size)
    def call(self, x_input, skip_input, training=False):
        if self.zoom == True:
            x = self.cropping(skip_input, training=training)
            x = self.x_path(x, training=training)
        else:
            x = self.x_path(skip_input, training=training)
        g = self.g_path(x_input, training=training)
        adding = Add()([x, g])
        x_g = self.activation_relu(adding)
        x_g = self.psi(x_g, training=training)
        x_g = self.activation_sigmoid(x_g)
        x_g = self.upsampling(x_g, training=training)
        if self.zoom == True:
            skip_zoom = self.cropping(skip_input, training=training)
            x_out = Multiply()([x_g, skip_zoom])
        else:
            x_out = Multiply()([x_g, skip_input])
        return x_out
    def get_config(self):
        config = super(AttentionGate, self).get_config()
        config.update({
            'channels': self.channels,
            'size': self.size,
            'zoom': self.zoom,
            'zoom_size': self.zoom_size,
            'g_path_config': self.g_path.get_config(),
            'x_path_config': self.x_path.get_config(),
            'psi_config': self.psi.get_config(),
            'activation_relu': 'relu',
            'activation_sigmoid': 'sigmoid',
            'upsampling_config': self.upsampling.get_config(),
            'cropping_config': self.cropping.get_config() if self.zoom else None,
        })
        return config
    @classmethod
    def from_config(cls, config):
        g_path_config = config.pop('g_path_config')
        x_path_config = config.pop('x_path_config')
        psi_config = config.pop('psi_config')
        upsampling_config = config.pop('upsampling_config')
        cropping_config = config.pop('cropping_config', None)
        g_path = Conv2D.from_config(g_path_config)
        x_path = Conv2D.from_config(x_path_config)
        psi = Conv2D.from_config(psi_config)
        upsampling = UpSampling2D.from_config(upsampling_config)
        cropping = CroppingLayer.from_config(cropping_config) if cropping_config else None
        instance = cls(**config)
        instance.g_path = g_path
        instance.x_path = x_path
        instance.psi = psi
        instance.activation_relu = Activation(config['activation_relu'])
        instance.activation_sigmoid = Activation(config['activation_sigmoid'])
        instance.upsampling = upsampling
        instance.cropping = cropping
        return instance

class DecodingBlock(Layer):
    def __init__(self, channels, block_type='standard', use_attention_gate=False, zoom_size=4):
        super(DecodingBlock, self).__init__()
        self.channels = channels
        self.block_type = block_type
        self.use_attention_gate = use_attention_gate
        self.zoom_size = zoom_size
        self.deconv = Conv2DTranspose(channels, kernel_size=3, strides=2, padding='same')
        self.conv_1 = CNNLayer(channels)
        self.conv_2 = CNNLayer(channels)
        self.cropping = CroppingLayer(self.zoom_size)
        if self.block_type == 'residual':
            self.initial_conv = CNNLayer(channels)
        if self.use_attention_gate == True:
            self.attention_gate = AttentionGate((channels * 2),zoom_size=self.zoom_size)
        self.optional_block = CNNBlock((channels*2),block_type=self.block_type, pool=False)
    def call(self, input_tensor, con_list, training=False):
        x_skip_list = []
        x = self.deconv(input_tensor, training=training)
        if self.use_attention_gate == True:
            for con in con_list:
                x_skip = self.attention_gate(input_tensor, con, training=training)
                x_skip_list.append(x_skip)
        else:
            for con in con_list:
                x_skip_zoom = self.cropping(con)
                x_skip_list.append(x_skip_zoom)
        x_skip_concat = Concatenate(axis=-1)(x_skip_list)
        x = Concatenate()([x, x_skip_concat])
        if self.block_type == 'residual':
            if len(con_list) > 1:
                x = self.optional_block(x,training=training)
            x = self.initial_conv(x, training=training)
            x_1 = self.conv_1(x, training=training)
            x_1 = self.conv_2(x_1, training=training)
            x = Add()([x, x_1])
        elif self.block_type == 'standard':
            if len(con_list) > 1:
                x = self.optional_block(x,training=training)
            x = self.conv_1(x, training=training)
            x = self.conv_2(x, training=training)
        else:
            print('spelling error')
        return x
    def get_config(self):
        config = super(DecodingBlock, self).get_config()
        config.update({
            'channels': self.channels,
            'block_type': self.block_type,
            'use_attention_gate': self.use_attention_gate,
            'zoom_size': self.zoom_size,
            'attention_gate_config': self.attention_gate.get_config() if self.use_attention_gate else None,
            'deconv_config': self.deconv.get_config(),
            'conv_1_config': self.conv_1.get_config(),
            'conv_2_config': self.conv_2.get_config(),
            'cropping_config': self.cropping.get_config(),
            'initial_conv_config': self.initial_conv.get_config() if self.block_type == 'residual' else None,
            'optional_block_config': self.optional_block.get_config(),
        })
        return config
    @classmethod
    def from_config(cls, config):
        deconv_config = config.pop('deconv_config')
        conv_1_config = config.pop('conv_1_config')
        conv_2_config = config.pop('conv_2_config')
        initial_conv_config = config.pop('initial_conv_config', None)
        attention_gate_config = config.pop('attention_gate_config', None)
        cropping_config = config.pop('cropping_config')
        optional_block_config = config.pop('optional_block_config')
        deconv = Conv2DTranspose.from_config(deconv_config)
        conv_1 = CNNLayer.from_config(conv_1_config)
        conv_2 = CNNLayer.from_config(conv_2_config)
        initial_conv = CNNLayer.from_config(initial_conv_config) if initial_conv_config else None
        attention_gate = AttentionGate.from_config(attention_gate_config) if attention_gate_config else None
        cropping = CroppingLayer.from_config(cropping_config)
        optional_block = CNNBlock.from_config(optional_block_config)

        instance = cls(**config)
        instance.deconv = deconv
        instance.conv_1 = conv_1
        instance.conv_2 = conv_2
        instance.initial_conv = initial_conv
        instance.attention_gate = attention_gate
        instance.cropping = cropping
        instance.optional_block = optional_block
        return instance


if __name__ == "__main__":

    # some test for some blocks
    gis_high_2 = keras.Input(shape=(32, 32, 32))
    gis_low = keras.Input(shape=(8, 8, 512))

    layer_2 = AttentionGate(64, zoom_size=8)


    # Example usage of the class
    input_shape = (2, 128, 128, 64)  # Time steps = 2
    inputs = Input(shape=input_shape)
    residual_block = ResidualConvLSTMBlock(filters=64)
    outputs = residual_block(inputs)

    model = Model(inputs, outputs)
    model.summary()


