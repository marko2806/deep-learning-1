# Deep Learning Laboratory Exercise: Convolutional Neural Networks with PyTorch and Custom Gradient Backpropagation

This laboratory exercise focuses on the implementation and application of Convolutional Neural Networks (ConvNets or CNNs) for image classification using both PyTorch and custom gradient backpropagation.

## Requirements

- Python 3.x
- PyTorch
- Numpy
- Matplotlib

## Datasets

The datasets used in this exercise are the [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. Both datasets contain images of handwritten digits (MNIST) and 10 different classes of objects (CIFAR), respectively.

## Implementation

The implementation of the ConvNet is done using PyTorch for one variant, and using custom gradient backpropagation for the other. The model architecture follows a typical ConvNet structure without dense layers:
- Convolutional layer
- Max pooling layer
- Flattening layer

## Usage

The python scripts contain the code to train and evaluate the models on the MNIST and CIFAR datasets. Simply run the scripts to train the models and evaluate the results.

## Results

The trained models achieved an accuracy of XX% on the test sets. The results can be visualized in the generated accuracy and loss plots.

## Conclusion

In this laboratory exercise, we have successfully implemented and trained ConvNets for image classification using both PyTorch and custom gradient backpropagation without dense layers. The models achieved good accuracy results, demonstrating the versatility and power of ConvNets in image classification tasks.
