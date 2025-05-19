# Temporal Convolutional Network (TCN) in Keras

This repository implements a **Temporal Convolutional Network (TCN)** for binary sequence classification tasks, using **causal convolutions**, **dilated convolutions**, and **skip connections** to capture long-range temporal dependencies. It is one page, and can be adjusted based on your needs.

## Features

- **Causal Convolutions**: Ensures temporal causality.
- **Dilated Convolutions**: Increases receptive field efficiently.
- **Skip Connections**: Aggregates features across layers.

## Installation

Install the necessary dependencies (plus your choice of Keras supported backends):

```bash
pip install keras
```
```python
from tcn import tcn

# Input shape: (timesteps, features)
num_inputs = (100, 1)  # Example

model = tcn(num_inputs, channels_list=[64, 128, 256], kernel=3, dropout=0.2, repeat=2, skip=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
predictions = model.predict(x_test)

```

## Citation

If you use this code, please cite:
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *Modeling Long-Range Temporal Dependencies with Dilated Convolutions*. [arXiv:1803.01271](https://arxiv.org/abs/1803.01271).
- Philippe Remy. *keras-tcn* [GitHub](https://github.com/philipperemy/keras-tcn/blob/master/tcn/tcn.py).

## License
  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Attribution

This documentation was written with the help of [ChatGPT](https://openai.com/chatgpt), an AI language model by OpenAI.

