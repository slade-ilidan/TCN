import keras
from keras import layers

@keras.saving.register_keras_serializable(package='TCN')
class TCNBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.0, name='TCNBlock', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.norm1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.norm2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()
        self.dropout2 = layers.Dropout(dropout_rate)

        self.downsample = None
        self.add = layers.Add()

    def build(self, input_shape):
        if input_shape[-1] != self.conv1.filters:
            self.downsample = layers.Conv1D(self.conv1.filters, kernel_size=1, padding='same')
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.act1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.dropout2(x, training=training)

        res = inputs if self.downsample is None else self.downsample(inputs)
        return self.add([x, res])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.conv1.filters,
            'kernel_size': self.conv1.kernel_size[0],
            'dilation_rate': self.conv1.dilation_rate[0],
            'dropout_rate': self.dropout1.rate
        })
        return config

@keras.saving.register_keras_serializable(package='TCN')
class TCN(keras.layers.Layer):
    def __init__(self, num_blocks=4, filters=64, kernel_size=3, dropout_rate=0.2, num_classes=1, name='TCN', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tcn_blocks = [
            TCNBlock(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=2**i,
                dropout_rate=dropout_rate
            ) for i in range(num_blocks)
        ]
        self.global_pool = layers.GlobalAveragePooling1D(name='GAP')
        self.output_layer = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax', name='Output')

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        for block in self.tcn_blocks: x = block(x, training=training)
        x = self.global_pool(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_blocks': len(self.tcn_blocks),
            'filters': self.tcn_blocks[0].conv1.filters,
            'kernel_size': self.tcn_blocks[0].conv1.kernel_size[0],
            'dropout_rate': self.tcn_blocks[0].dropout1.rate,
            'num_classes': self.output_layer.units
        })
        return config

