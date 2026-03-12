# Deep Learning Project 1: Convolutional Neural Networks

## About the Project
The objective of this project is to perform image classification on the CINIC-10 dataset using convolutional neural networks (CNNs). This project is realized as a part of the Deep Learning course at the Faculty of Mathematics and Information Science, Warsaw University of Technology.

The research includes comparing different network architectures, evaluating hyperparameter changes and optimization algorithms, exploring regularization techniques, and observing how models perform with severely restricted data (Few-Shot Learning).

## Authors
* [**Norbert Frydrysiak**](https://github.com/fantasy2fry)
* [**Michał Kukla**](https://github.com/mickuk)

## Dataset
The project utilizes the **CINIC-10** dataset.

## Network Architectures
To evaluate performance and computational efficiency, the following models were implemented and tested:
* **Custom CNN**: A convolutional neural network designed and implemented entirely from scratch to serve as a foundational baseline.
* **VGG-11**: A classic, deep architecture trained from scratch. It serves as an environment for testing regularization techniques, specifically Dropout.
* **ResNet-34**: A modern standard utilizing skip-connections. We perform fine-tuning on a pre-trained version of this model.
* **MobileNetV2**: An ultra-lightweight network chosen to address potential hardware constraints and evaluate the trade-off between model size and accuracy (fine-tuned).
* **EfficientNet-B0** *(Optional)*: A highly optimized, state-of-the-art baseline model for fine-tuning, resources permitting.

## Conducted Experiments
As part of the research process, we investigate the influence of the following factors on the models' performance:

1. **Training Process & Optimizers**:
   * Experimenting with different values of Learning Rate (including schedulers) and Batch Size.
   * Comparing standard SGD (Stochastic Gradient Descent) against adaptive optimizers like Adam.
2. **Regularization**:
   * Testing the impact of Dropout layers.
   * Applying ElasticNet penalty (L1 + L2) or Weight Decay to prevent overfitting.
3. **Data Augmentation**:
   * **Standard Operations**: Random Horizontal Flip, Random Crop, and Random Rotation.
   * **Advanced Techniques**: The Cutout augmentation method.
4. **Few-Shot Learning & Dataset Reduction**:
   * Drastically reducing the size of the training set to test performance with restricted data.
   * **Transfer Learning**: Freezing the feature extraction layers of a pre-trained ResNet-34 and training only the final classifier.
   * Incorporating **Contrastive Learning** to learn robust representations from limited data.
   * Comparing results obtained from the reduced dataset with those trained on the entire dataset.

## Reproducibility
To ensure full reproducibility of our experiments, we set a constant seed for the random number generator at the beginning of our workflow.

## How to Run

```bash
# Clone the repository
git clone https://github.com/fantasy2fry/CNN-Image-Classification-CINIC10
cd CNN-Image-Classification-CINIC10

# Create virtual environment
python3.12.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt