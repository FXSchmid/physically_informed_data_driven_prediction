import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from architecture import deeplearning_blocks
from keras.layers import Concatenate, Add, ConvLSTM2D

class SequenenceNetwork(Model):
    def __init__(self, channels):
        super(SequenenceNetwork, self).__init__()
        self.channels=channels

        self.encoding_gis_1 = deeplearning_blocks.CNNBlock(self.channels[0], block_type='residual',pool=True)
        self.encoding_gis_2 = deeplearning_blocks.CNNBlock(self.channels[1], block_type='residual', pool=True)
        self.encoding_gis_3 = deeplearning_blocks.CNNBlock(self.channels[2], block_type='residual', pool=True)
        self.encoding_gis_4 = deeplearning_blocks.CNNBlock(self.channels[3], block_type='residual', pool=True)

        self.encoding_water_1 = deeplearning_blocks.ResidualConvLSTMBlock(self.channels[0])
        self.encoding_water_2 = deeplearning_blocks.ResidualConvLSTMBlock(self.channels[1])
        self.encoding_water_3 = deeplearning_blocks.ResidualConvLSTMBlock(self.channels[2])
        self.encoding_water_4 = deeplearning_blocks.ResidualConvLSTMBlock(self.channels[3])
        self.encoding_water_next_frame = ConvLSTM2D(filters=self.channels[3], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', return_sequences=False)

        self.encoding_rain_1 = deeplearning_blocks.CNNBlock(self.channels[1], block_type='residual', pool=False)
        self.encoding_rain_2 = deeplearning_blocks.CNNBlock(self.channels[2], block_type='residual', pool=False)
        self.encoding_rain_3 = deeplearning_blocks.CNNBlock(self.channels[3], block_type='residual', pool=False)
        self.encoding_rain_4 = deeplearning_blocks.CNNBlock(self.channels[4], block_type='residual', pool=False)

        self.center = deeplearning_blocks.CNNLayer(self.channels[4])

        self.cropping = deeplearning_blocks.CroppingLayer(4)

        self.decoder_1 = deeplearning_blocks.DecodingBlock(self.channels[3], block_type='residual', use_attention_gate=True, zoom_size=8)
        self.decoder_2 = deeplearning_blocks.DecodingBlock(self.channels[2], block_type='residual', use_attention_gate=True, zoom_size=16)
        self.decoder_3 = deeplearning_blocks.DecodingBlock(self.channels[1], block_type='residual', use_attention_gate=True, zoom_size=32)
        self.decoder_4 = deeplearning_blocks.DecodingBlock(self.channels[0], block_type='residual', use_attention_gate=False, zoom_size=64)

        self.outputs_1 = deeplearning_blocks.CNNLayer(32,activation='relu',use_bias=False, use_batch_norm=True)
        self.outputs_2 = deeplearning_blocks.CNNLayer(11, activation='linear',use_bias=True, use_batch_norm=False)

    def call(self, input_gis, input_water, input_rain, training=False):

        x, skip_gis_1 = self.encoding_gis_1(input_gis, training=training)
        x, skip_gis_2 = self.encoding_gis_2(x, training=training)
        x, skip_gis_3 = self.encoding_gis_3(x, training=training)
        x, skip_gis_4 = self.encoding_gis_4(x, training=training)

        xs, skip_water_1 = self.encoding_water_1(input_water, training=training)
        xs, skip_water_2 = self.encoding_water_2(xs, training=training)
        xs, skip_water_3 = self.encoding_water_3(xs, training=training)
        xs, skip_water_4 = self.encoding_water_4(xs, training=training)
        xs = self.encoding_water_next_frame(xs, training=training)

        xr = self.encoding_rain_1(input_rain, training=training)
        xr = self.encoding_rain_2(xr, training=training)
        xr = self.encoding_rain_3(xr, training=training)
        xr = self.encoding_rain_4(xr, training=training)

        skiconnections_4 = [skip_gis_4, skip_water_4]
        skiconnections_3 = [skip_gis_3, skip_water_3]
        skiconnections_2 = [skip_gis_2, skip_water_2]
        skiconnections_1 = [skip_gis_1, skip_water_1]

        x = Concatenate()([x, xs])
        x = Add()([x,xr])
        x = self.center(x,training=training)
        x = self.cropping(x, training=training)

        x = self.decoder_1(x, skiconnections_4,training=training)
        x = self.decoder_2(x, skiconnections_3, training=training)
        x = self.decoder_3(x, skiconnections_2, training=training)
        x = self.decoder_4(x, skiconnections_1, training=training)

        x = self.outputs_1(x,training=training)
        y = self.outputs_2(x, training=training)


        return y

    def build_graph(self):
        gis = keras.Input(shape=(256, 256, 8))
        water = keras.Input(shape=(2 ,256, 256, 1))
        rain = keras.Input(shape=(16, 16, 22)) 
        forecast_model = keras.Model(inputs=[gis,water,rain], outputs=self.call(gis,water,rain))
        return forecast_model
    def get_config(self):
        config = super(SequenenceNetwork, self).get_config()
        config.update({
            'channels': self.channels,
            'encoding_gis_1_config': self.encoding_gis_1.get_config(),
            'encoding_gis_2_config': self.encoding_gis_2.get_config(),
            'encoding_gis_3_config': self.encoding_gis_3.get_config(),
            'encoding_gis_4_config': self.encoding_gis_4.get_config(),
            'encoding_water_1_config': self.encoding_water_1.get_config(),
            'encoding_water_2_config': self.encoding_water_2.get_config(),
            'encoding_water_3_config': self.encoding_water_3.get_config(),
            'encoding_water_4_config': self.encoding_water_4.get_config(),
            'encoding_rain_1_config': self.encoding_rain_1.get_config(),
            'encoding_rain_2_config': self.encoding_rain_2.get_config(),
            'encoding_rain_3_config': self.encoding_rain_3.get_config(),
            'encoding_rain_4_config': self.encoding_rain_4.get_config(),
            'center_config': self.center.get_config(),
            'cropping_config': self.cropping.get_config(),
            'decoder_1_config': self.decoder_1.get_config(),
            'decoder_2_config': self.decoder_2.get_config(),
            'decoder_3_config': self.decoder_3.get_config(),
            'decoder_4_config': self.decoder_4.get_config(),
            'outputs_1_config': self.outputs_1.get_config(),
            'outputs_2_config': self.outputs_2.get_config(),
        })
        return config
    @classmethod
    def from_config(cls, config):
        channels = config.pop('channels')
        encoding_gis_1_config = config.pop('encoding_gis_1_config')
        encoding_gis_2_config = config.pop('encoding_gis_2_config')
        encoding_gis_3_config = config.pop('encoding_gis_3_config')
        encoding_gis_4_config = config.pop('encoding_gis_4_config')
        encoding_water_1_config = config.pop('encoding_water_1_config')
        encoding_water_2_config = config.pop('encoding_water_2_config')
        encoding_water_3_config = config.pop('encoding_water_3_config')
        encoding_water_4_config = config.pop('encoding_water_4_config')
        encoding_rain_1_config = config.pop('encoding_rain_1_config')
        encoding_rain_2_config = config.pop('encoding_rain_2_config')
        encoding_rain_3_config = config.pop('encoding_rain_3_config')
        encoding_rain_4_config = config.pop('encoding_rain_4_config')
        center_config = config.pop('center_config')
        cropping_config = config.pop('cropping_config')
        decoder_1_config = config.pop('decoder_1_config')
        decoder_2_config = config.pop('decoder_2_config')
        decoder_3_config = config.pop('decoder_3_config')
        decoder_4_config = config.pop('decoder_4_config')
        outputs_1_config = config.pop('outputs_1_config')
        outputs_2_config = config.pop('outputs_2_config')

        instance = cls(channels)
        instance.encoding_gis_1 = deeplearning_blocks.CNNBlock.from_config(encoding_gis_1_config)
        instance.encoding_gis_2 = deeplearning_blocks.CNNBlock.from_config(encoding_gis_2_config)
        instance.encoding_gis_3 = deeplearning_blocks.CNNBlock.from_config(encoding_gis_3_config)
        instance.encoding_gis_4 = deeplearning_blocks.CNNBlock.from_config(encoding_gis_4_config)
        instance.encoding_water_1 = deeplearning_blocks.CNNBlock.from_config(encoding_water_1_config)
        instance.encoding_water_2 = deeplearning_blocks.CNNBlock.from_config(encoding_water_2_config)
        instance.encoding_water_3 = deeplearning_blocks.CNNBlock.from_config(encoding_water_3_config)
        instance.encoding_water_4 = deeplearning_blocks.CNNBlock.from_config(encoding_water_4_config)
        instance.encoding_rain_1 = deeplearning_blocks.CNNLayer.from_config(encoding_rain_1_config)
        instance.encoding_rain_2 = deeplearning_blocks.CNNLayer.from_config(encoding_rain_2_config)
        instance.encoding_rain_3 = deeplearning_blocks.CNNLayer.from_config(encoding_rain_3_config)
        instance.encoding_rain_4 = deeplearning_blocks.CNNLayer.from_config(encoding_rain_4_config)
        instance.center = deeplearning_blocks.CNNLayer.from_config(center_config)
        instance.cropping = deeplearning_blocks.CroppingLayer.from_config(cropping_config)
        instance.decoder_1 = deeplearning_blocks.DecodingBlock.from_config(decoder_1_config)
        instance.decoder_2 = deeplearning_blocks.DecodingBlock.from_config(decoder_2_config)
        instance.decoder_3 = deeplearning_blocks.DecodingBlock.from_config(decoder_3_config)
        instance.decoder_4 = deeplearning_blocks.DecodingBlock.from_config(decoder_4_config)
        instance.outputs_1 = deeplearning_blocks.CNNLayer.from_config(outputs_1_config)
        instance.outputs_2 = deeplearning_blocks.CNNLayer.from_config(outputs_2_config)

        return instance


if __name__ == "__main__":

    dot_img_file = r'...' # for storing the plot
    channels = [64,128,256,512,1024]
    Model = SequenenceNetwork(channels)
    plotting = Model.build_graph()
    tf.keras.utils.plot_model(plotting, to_file=dot_img_file, show_shapes=True, expand_nested=True)
    print(plotting.summary())



