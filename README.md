# DAIN

A TensorFlow 2.x Implemention of Deep Adaptive Input Normalization for Time Series Using Keras API

**TL;DR: Here is a trainable neural layer for time series normalization.**

# Usage
```python
from tensorflow import keras
from dain import Dain

dim = 1000 # number of rows in a sample
n_features = 100 # number of features in a row

model = keras.Sequential()
model.add(Dain(dim, n_features))
model.add(...)

# training
# raw_features.shape == (batch_size, dim, n_features)
model.fit(raw_features, targets)
```

# Components
DAIN ([Passalis et al., 2019](https://arxiv.org/abs/1902.07892)) behaviors as z-score standardization with feature selection in a trainable/adaptive way. The authors claim the preprocessing layer generalizes preprocessing well for time series data with the key components:
 - Adaptive Averaging (updates on mean value)
 - Adaptive Scaling (updates on standard diavation)
 - Gating (suppresses unrelated features)

# Future Work
In the [original pytorch version](https://github.com/passalis/dain/), a learning rate multiplier for `Adam` optimizer is used to apply different fine-tuned learning rates on the DAIN weights. This is not implemented in the TensorFlow Keras Implemention. As an alternative, please refer to a `keras-contrib` implementation [here](https://github.com/stante/keras-contrib/blob/feature-lr-multiplier/keras_contrib/optimizers/lr_multiplier.py).
