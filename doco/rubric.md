# 🧠 Project: Developing a Handwritten Digits Classifier with PyTorch

This project involves developing a neural network using PyTorch to classify handwritten digits using the MNIST dataset. Below is a breakdown of the rubric criteria, which you should aim to meet or exceed.

---

## 📦 Section 1: Data Loading and Exploration

### ✅ Transform data for use in neural networks
- Data is preprocessed and converted to a tensor using `transforms.ToTensor()` from `torchvision.transforms` or manually with `torch.Tensor`.

### ✅ Use `DataLoader` to input data into the neural network
- A `DataLoader` object is created for both training and test sets using datasets loaded from `torchvision.datasets`.

### ✅ Explore datasets and describe their properties
- Code demonstrates the size and shape of both training and test datasets.
- At least one image from the dataset is visualised using `imshow()` or a similar method.

### ✅ Provide justification for preprocessing
- Markdown or code comments explain why preprocessing is necessary (e.g. flattening, tensor conversion, normalization).

---

## 🧠 Section 2: Model Design and Training

### ✅ Build a neural network for image classification
- A PyTorch `Model` or `nn.Sequential` class is defined with:
  - At least **two hidden layers**
  - A forward method that outputs prediction probabilities using **softmax**

### ✅ Select an appropriate loss function
- A classification-appropriate loss function (e.g. `nn.CrossEntropyLoss`) is implemented.

### ✅ Define an optimizer
- An optimizer from `torch.optim` (e.g. `SGD`, `Adam`) is used to minimise the loss and update weights.

---

## 📊 Section 3: Model Testing and Evaluation

### ✅ Use DataLoader and holdout set for testing
- The test `DataLoader` is used to:
  - Get predictions from the trained network
  - Compare predictions to ground truth labels

### ✅ Optimise model to achieve at least 90% accuracy
- Hyperparameters are tuned (e.g. learning rate, architecture, batch size).
- The model achieves **≥ 90% classification accuracy** on the test set.

### ✅ Save trained model
- The model is saved using `torch.save()` so it can be reloaded later.

---

## 🌟 Suggestions to Make Your Project Stand Out

- ✅ Implement a validation set to monitor performance at each epoch
- ✅ Upgrade to a **Convolutional Neural Network (CNN)** for better accuracy
- ✅ Contextualise performance by comparing your results to benchmarks like [Yann LeCun’s MNIST page](http://yann.lecun.com/exdb/mnist/)

---
