# PyANN Examples

Each script runs on its own: `python examples/<name>.py` (add `--help` for options).
The Fashion-MNIST examples download the dataset (about 30 MB) into `examples/data/`
the first time, or take the Kaggle CSV files with `--train-csv` / `--test-csv`.
Add `--gpu` to train on Metal (macOS) or Vulkan (Linux, Windows) when available.

Timings below are from an Apple-silicon Mac (Metal GPU for the image models).

## No download needed

| Example | Shows | Runs in |
|---|---|---|
| [`logic_gates.py`](logic_gates.py) | AND/OR/NAND/XOR with plain Python lists; why XOR needs a hidden layer; PIKCHR diagram export | < 1 s |
| [`digits_5x7.py`](digits_5x7.py) | Recognizing 5x7 pixel digits; training with salt-and-pepper `set_input_noise()` for robustness to flipped pixels | < 2 s |
| [`regression_activations.py`](regression_activations.py) | Curve fitting with a linear output and MSE; compares sigmoid, tanh, softsign, ReLU, leaky ReLU and GELU; validation data, `EarlyStopping(restore_best=True)`, `CosineAnnealing` | < 1 s |
| [`online_learning.py`](online_learning.py) | `train_on_batch()` on a stream whose labelling rule drifts; a frozen model falls to about 50% while the online one stays above 90%; save, reload and keep learning | < 1 s |

## Fashion-MNIST

| Example | Shows | Result |
|---|---|---|
| [`mnist_fashion.py`](mnist_fashion.py) | A dense MLP classifier, `ExponentialDecay`, saving and ONNX export | about 85% (784-32-10) |
| [`fashion_cnn.py`](fashion_cnn.py) | `Conv2D`, `MaxPool2D`, `Flatten`; a validation split with early stopping; per-class accuracy and common mistakes; ONNX round trip | about 89% test accuracy, 13 s on Metal |
| [`fashion_denoise.py`](fashion_denoise.py) | A convolutional denoising autoencoder with `UpSample2D`: `fit(x, x)` with Gaussian, masking or salt-and-pepper input noise; writes a PNG of clean, noisy and denoised images | PSNR 12.8 dB to 19.2 dB, 21 s on Metal |
| [`fashion_anomaly.py`](fashion_anomaly.py) | A 32-value `Dense` bottleneck and `Reshape` back to an image; trained without one class, it flags that class by reconstruction error | ROC AUC about 0.90 for the unseen class |
| [`deploy_int8.py`](deploy_int8.py) | `quantize()` to int8, float32 vs int8 accuracy, C header export for microcontrollers; compiles the header with your C compiler and checks it matches `predict_int8()` bit for bit | int8 within 0.2 points of float32 at a quarter of the memory |
| [`hyperparameter_search.py`](hyperparameter_search.py) | Random search vs TPE over learning rate, batch size, depth, widths, activations and dropout; retrains the winner on all data | about 88% test accuracy, 20 s |

[`example_data.py`](example_data.py) holds the shared helpers: the dataset loader, backend setup, ASCII and PNG image output.

`tests/test_examples.py` runs every example on small synthetic data, so they keep working as the library changes.
