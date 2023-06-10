# Evolutionary Strategies for Continuous Learning

This repository contains the algorithms developed and presented in the Master thesis entitled 
**Evolving Deep Neural Networks for Continuous Learning: Addressing Challenges and Adapting to Changing Data Conditions without Catastrophic Forgetting**. The thesis was completed as a requirement for the APPMAS course at the University of Stavanger.

The thesis explores and implements evolutionary strategies for continuous learning, a cutting-edge approach that enables adaptive and lifelong learning in dynamic environments. 

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Project structure](#project-structure)
- [How to navigate the repository](#how-to-navigate-the-repository)
- [Acknowledgements](#acknowledgements)


## Overview

Continuous learning is crucial in domains where data distributions change over time or where new concepts emerge. This repository aims to provide a comprehensive collection of novel approach to address the challenges of continuous learning. By leveraging evolutionary strategies, the goal is to facilitate the development of intelligent systems that can learn and adapt continuously in dynamic environments.


## Data
This approach was evaluates using 3 different benchmark datasets.

- MNIST is a widely-used dataset in the field of machine learning and computer vision. It consists of a collection of 70,000 grayscale images of handwritten digits, each with a size of 28x28 pixels.This dataset can be obtained via TensorFlow's `tf.keras.datasets.mnist.load_data()` command.
- Fashion-MNIST provides a training set of 60,000 images and a separate test set of 10,000 images. This dataset can be obtained from the official repository maintained by Zalando Research or can be accessed by via TensorFlow's `tf.keras.datasets.fashion_mnist.load_data()` command.
- CIFAR-10 is benchmark dataset that consists of 60,000 color images, each with a size of 32x32 pixels, categorized into 10 different classes. It can be accessed via TensorFlow's `tf.keras.datasets.cifar10.load_data()` command.

Please note that the MNIST dataset is used twice in this repository. 

## Installation

To use the evolutionary strategies for continuous learning framework, follow these steps:

1. Clone this repository: `git clone [https://github.com/your-username/your-repository.git](https://github.com/ata-bruna/thesis_es_continuous_learning)`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Explore the code examples in the repository to get started.

## Project structure

The project is organized as follows:

``` sh
📦es_continuous_learning
 ┣ 📂confusion_matrix *
 ┃ ┣ 📂CIFAR-10 *
 ┃ ┣ 📂Fashion-MNIST *
 ┃ ┣ 📂MNIST *
 ┃ ┗ 📂MNIST-CNN *
 ┣ 📂figures *
 ┃ ┣ 📂CIFAR-10 *
 ┃ ┣ 📂Fashion-MNIST *
 ┃ ┣ 📂MNIST *
 ┃ ┣ 📂MNIST-CNN *
 ┃ ┗ 📂Summary *
 ┣ 📂results *
 ┃ ┗ 📜results.csv
 ┣ 📜.gitignore
 ┣ 📜cifar_10.py
 ┣ 📜fashion_mnist.py
 ┣ 📜helper_functions.py
 ┣ 📜mnist_cnn.py 
 ┣ 📜mnist.py
 ┣ 📜plotting_results.ipynb
 ┣ 📜README.md
 ┗ 📜requirements.txt
```

Please note that the folders marked with `*` need to be created in order to save the pictures necessary to evaluate the models'performance.

## How to navigate the repository

Each `.py` can be run independently. Change `STRATIFY` and `HIDE` variables to `True` or `False` in line 26 and 27 of each the `.py` files to apply the evolutionary strategies for different data splits while optionally removing one class.


-  `helper_functions.py` contains helper functions used to perform evolutionary strategies or to convert the data into a specific format used by the models.
-  `CIFAR-10.py` contains the approach applied to the CIFAR-10 dataset.
-  `fashion_mnist.py` contains the approach applied to the Fashion-MNIST dataset.
-  `mnist_cnn.py` contains the approach applied to the MNIST-CNN dataset.
-  `mnist.py`contains the approach applied to the MNIST dataset.

## Acknowledgements

We would like to express our gratitude to the open-source community for their valuable contributions and to the authors of the referenced papers and datasets that have enabled advancements in the field of continuous learning through evolutionary strategies.

---

We hope you find the Evolutionary Strategies for Continuous Learning repository valuable in your research and development endeavors. If you have any questions or suggestions, please feel free to reach out to us.

Happy continuous learning!
