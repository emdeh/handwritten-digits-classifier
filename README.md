
# 🧠 MNIST OCR Classifier (PyTorch)

This project builds a neural network using PyTorch to classify handwritten digits from the MNIST dataset — a proof of concept for future optical character recognition (OCR) systems.

This is a project as part of a Deep Learning short course.

## 📌 Objectives

- Load and preprocess MNIST dataset
- Build a multi-layer neural network classifier using PyTorch
- Train, evaluate, and optimise the model to achieve ≥ 90% accuracy
- Save the trained model for reuse
- Explore good software practices (versioning, env setup, doco)

## ✅ Task Checklist

- [x] Loaded the dataset from `torchvision.datasets`.
- [ ] Used transforms or other `PyTorch` methods to convert the data to tensors, normalise, and flatten the data.
- [ ] Created a `DataLoader` for my dataset.
- [ ] Visualised the dataset using the provided function.
- [ ] Used either training data or inverting any normalisation and flattening OR a second `DataLoader` without any normalisation or flattening.
- [ ] Provided a brief justification of any necessary preprocessing steps or why no preprocessing is needed.
- [ ] Used `PyTorch` to build a neural network to predict the class of each given input image.
- [ ] Created an optimiser to update my network's weights.
- [ ] Used the training `DataLoaderto` train my neural network.
- [ ] Tuned my model hyperparameters and network architecture with at least 90% accuracy on the test set.
- [ ] Used torch.saveto save my trained model.

### Project Rubric

See [`rubric.md`](/doco/rubric.md)

## 🧪 Environment Setup

See [`setup_env.md`](/doco/setup_env.md) for full instructions.

## 🚀 Future Enhancements

- Add validation set for early stopping
- Replace dense network with a CNN
- Enable remote training for AMD GPU systems (WIP)

## 📂 File Structure

```bash
├── MNIST_Handwritten_Digits-STARTER.ipynb
├── README.md
├── doco
│   ├── rubric.md
│   └── setup_env.md
├── environment.yml
└── requirements.txt
```

## 📚 References

- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
