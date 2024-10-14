# MarsGT: Multimodal Graph Transformer for Single-Cell Data

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Introduction

**MarsGT** is a Multimodal Graph Transformer designed for single-cell data analysis. It leverages gene expression (RNA) data to construct and train graph neural networks (GNNs) that capture the intricate relationships between genes and cells. By focusing solely on RNA matrices, MarsGT simplifies the modeling process while maintaining robust performance for downstream tasks such as clustering and classification.

## Features

- **Graph Neural Network Integration**: Utilizes GNNs to model interactions between genes and cells.
- **Label Smoothing**: Implements label smoothing to enhance model generalization.
- **Efficient Training and Prediction**: Optimized for handling large single-cell datasets.
- **Flexible Architecture**: Easily adaptable for various downstream tasks.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/StickTaTa/ScFormer.git
    cd MarsGT
    ```

2. **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- SciPy
- tqdm

**Note**: Ensure that you have the appropriate CUDA version installed if you intend to use GPU acceleration.

## Example

see https://github.com/StickTaTa/ScFormer/blob/main/01_cell_graph.ipynb

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

