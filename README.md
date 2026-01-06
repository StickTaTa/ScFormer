#  scFormer: A heterogeneous graph transformer-based method for rare cell identification in single-cell expression data

![img.png](Fig/img.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Introduction

**ScFormer** is a graph-based computational framework designed to detect rare cell populations by encoding biological priors directly
into the underlying topology. Rather than treating all expressed genes equivalently, scFormer is motivated by the premise that broadly
expressed genes provide limited discriminatory power for resolving rare states. Instead, it prioritizes genes exhibiting high
cell-specific expression, treating them as informative anchors for distinguishing rare populations.

## Features

- **Graph Neural Network Integration**: Utilizes GNNs to model interactions between genes and cells.
- **Label Smoothing**: Implements label smoothing to enhance model generalization.
- **Efficient Training and Prediction**: Optimized for handling large single-cell datasets.
- **Flexible Architecture**: Easily adaptable for various downstream tasks.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/StickTaTa/ScFormer.git
    cd scformer
    ```

2. **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install Dependencies**: You can check the dependencies and their versions in the `requirements.txt` file.

## Requirements

Our code was tested on a Windows platform with the following hardware specifications:
- **CPU**: Intel(R) Core(TM) i9-14900KF
- **GPU**: NVIDIA GeForce RTX 4090 D
- **CUDA version**: 11.8


- Python 3.8+
- PyTorch 2.1.0
- NumPy
- SciPy
- tqdm

**Note**: Ensure that you have the appropriate CUDA version installed if you intend to use GPU acceleration.

## Example

The following file contains a tutorial for the model:

- [01_Tutorial_example.ipynb](https://github.com/StickTaTa/ScFormer/blob/main/Tutorial/01_Tutorial_example.ipynb) contains a simple example using simulated data, including 500 cells.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.
