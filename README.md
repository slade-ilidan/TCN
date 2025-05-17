# Temporal Convolutional Network Classifier
https://arxiv.org/abs/1803.01271

Keras 3.0 API (Backends: PyTorch, TensorFlow, JAX)

```python
import keras
from tcn import TCN

# load data
X = ...
Y = ...

# model
inputs = keras.Input((X.shape[1:]))
outputs = TCN(num_blocks=4, filters=64, kernel_size=3, dropout_rate=0.2, num_classes=1) # num_blocks determine dilation (e.g. 4 then dilations up to 2**4)
model = keras.Model(inputs, outputs)
model.compile(optimizer, loss, metrics)
model.fit(X, Y, batch_size, epochs)
model.save('tcn.keras')
model = keras.saving.load_model('tcn.keras')
```

