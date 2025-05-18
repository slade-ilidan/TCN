import keras
from keras import layers

@keras.saving.register_keras_serializable(package='TCN')
class TCNBlock(layers.Layer):
    def __init__(self, filters, kernel, dilation, dropout, name='TCNBlock', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = layers.Conv1D(filters, kernel, padding='causal', dilation_rate=dilation)
        self.norm1 = layers.LayerNormalization()
        self.act1 = layers.ReLU()
        self.dropout1 = layers.Dropout(dropout)
        
        self.conv2 = layers.Conv1D(filters, kernel, padding='causal', dilation_rate=dilation)
        self.norm2 = layers.LayerNormalization()
        self.act2 = layers.ReLU()
        self.dropout2 = layers.Dropout(dropout)

        self.downsample = None
        self.add = layers.Add()

    def build(self, input_shape):
        if input_shape[-1] != self.conv1.filters:
            self.downsample = layers.Conv1D(self.conv1.filters, kernel_size=1, padding='same')
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout2(x, training=training)

        res = inputs if self.downsample is None else self.downsample(inputs)
        return self.add([x, res])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.conv1.filters,
            'kernel': self.conv1.kernel[0],
            'dilation': self.conv1.dilation[0],
            'dropout': self.dropout1.rate
        })
        return config

@keras.saving.register_keras_serializable(package='TCN')
class TCN(keras.layers.Layer):
    def __init__(self, blocks=1, filters=64, kernel=3, dropout=0.2, classes=1, name='TCN', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tcn_blocks = [
            TCNBlock(
                filters=filters,
                kernel=kernel,
                dilation=2**i,
                dropout=dropout
            ) for i in range(blocks)
        ]
        self.global_pool = layers.GlobalAveragePooling1D(name='GAP')
        self.output_layer = layers.Dense(classes, activation='sigmoid' if classes == 1 else 'softmax', name='Output')

    # def build(self, input_shape):
    #     super().build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        for block in self.tcn_blocks: x = block(x, training=training)
        x = self.global_pool(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'blocks': len(self.tcn_blocks),
            'filters': self.tcn_blocks[0].conv1.filters,
            'kernel': self.tcn_blocks[0].conv1.kernel[0],
            'dropout': self.tcn_blocks[0].dropout1.rate,
            'classes': self.output_layer.units
        })
        return config

