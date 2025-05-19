import keras

def tcn_block(in_channels, out_channels, kernel, dilation, dropout):
    x = keras.layers.Conv1D(out_channels, kernel_size=kernel, dilation_rate=dilation, padding='causal')(in_channels)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.SpatialDropout1D(dropout)(x)

    x = keras.layers.Conv1D(out_channels, kernel_size=kernel, dilation_rate=dilation, padding='causal')(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.SpatialDropout1D(dropout)(x)
    x_skip = x

    if in_channels.shape[-1] != out_channels: in_channels = keras.layers.Conv1D(out_channels, kernel_size=1)(in_channels)
    x = keras.layers.Add()([x, in_channels])
    x = keras.layers.ReLU()(x)
    return x, x_skip

def tcn(num_inputs, channels_list=[64], kernel=3, dropout=0.2, repeat=1, skip=True):
    inputs = keras.layers.Input(shape=num_inputs)
    x = inputs

    levels = len(channels_list)
    skip_connections = []
    for level, channels in enumerate(channels_list):
        dilation = 2**level
        level_input = x

        for _ in range(repeat):
            x, x_skip = tcn_block(x, channels, kernel, dilation, dropout)
            skip_connections.append(x_skip)

    if skip and skip_connections:
        x = keras.layers.Add()(skip_connections)

    x = keras.layers.GlobalAveragePooling1D()(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, output)
    return model

