# Temporal Convolutional Network Classifier
https://arxiv.org/abs/1803.01271

Keras 3.0 API (Backends: PyTorch, TensorFlow, JAX)

```python
import numpy as np

X = np.random.randn(1000, 256, 1) 
y = np.random.randint(0, 2, size=(1000, 1))

model = tcn(X.shape[1:], [64, 64, 64], repeat=2)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"],)
model.fit(X, y, epochs=10, batch_size=32)
```

