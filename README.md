# 2D Ising Model Classification with Convolutional Neural Networks

This repository contains an example Python script illustrating how to **simulate** the 2D Ising model using the **Metropolis Monte Carlo method** and subsequently **classify** the low-temperature (ordered) phase versus the high-temperature (disordered) phase using a simple **Convolutional Neural Network (CNN)** in PyTorch.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Key Features](#key-features)
- [References](#references)

---

## Overview
1. **Monte Carlo Simulation (Metropolis)**  
   A classical approach to generate realistic spin configurations for the 2D Ising model. Each spin configuration is labeled as:
   - **0 (ordered)** if \( T < T_c \)  
   - **1 (disordered)** if \( T > T_c \)  
   where \( T \) is the temperature and \( T_c \approx 2.269 \) is the critical temperature for the 2D Ising model.

2. **Deep Learning Classifier**  
   A simple CNN is trained in **PyTorch** to distinguish between the two phases. The script demonstrates:
   - Splitting the dataset into train/test sets.
   - Training for multiple epochs.
   - Tracking and plotting loss/accuracy curves.
   - Evaluating via a confusion matrix.

This workflow showcases how computational physics methods can be combined with modern machine learning techniques to classify phases in a toy physical system.

---

## Requirements
- **Python 3.7+**  
- **PyTorch** (preferably with CUDA support if available)
- **NumPy**
- **Matplotlib**

To install these with pip:
```bash
pip install torch torchvision numpy matplotlib
```

---

## Usage

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/YourUsername/YourRepoName.git
   cd YourRepoName
   ```

2. **Install Dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
   *(Or manually install via `pip install torch torchvision numpy matplotlib`.)*

3. **Run the Code**:
   ```bash
   python ising_cnn.py
   ```
   This will:
   - Generate spin configurations at specified temperatures using Metropolis Monte Carlo.
   - Train the CNN to classify ordered vs. disordered phases.
   - Plot training/validation losses and accuracies, then display a confusion matrix.

4. **Adjust Hyperparameters**  
   Open `ising_cnn.py` and modify variables such as:
   - `lattice_size` (e.g., 16x16)
   - `num_samples` (per temperature)
   - `temperatures` (list of temperatures to simulate)
   - `num_epochs` (epochs to train)
   - `n_sweeps` (number of Metropolis sweeps)

---

## Code Structure

- **`metropolis_single_flip`**  
  Performs a single site-flip attempt in the Metropolis algorithm.
- **`metropolis_ising`**  
  Initializes and thermalizes a spin configuration by calling `metropolis_single_flip` multiple times.
- **`Ising2DData`** (PyTorch `Dataset`)  
  - Generates or stores spin configurations at various temperatures.
  - Labels them based on whether \( T \) is below or above \( T_c \).
- **`SimpleCNN`**  
  - A compact convolutional network with two convolutional layers and two fully connected layers.
- **Training/Validation Functions** (`train_model`, `validate_model`)  
  - Typical PyTorch training and validation routines (forward pass, loss calculation, backprop).
- **Plotting/Analysis** (`plot_learning_curves`, `evaluate_and_plot_confusion`)  
  - Plots epoch-wise loss/accuracy and creates a confusion matrix on the test set.
- **`main`**  
  - Assembles the entire pipeline: data loading, model training, evaluation, and plotting.

---

## Key Features

1. **Physical Realism**  
   - The **Metropolis** method creates physically plausible spin configurations rather than random ones.
2. **PyTorch Integration**  
   - Direct usage of PyTorch’s `Dataset` and `DataLoader` for efficient batch processing.
3. **Easy Customization**  
   - Hyperparameters (e.g., `num_epochs`, `batch_size`) can be tweaked to experiment with different training regimes.
4. **Data Visualization**  
   - Training/validation curves and a confusion matrix are automatically displayed for quick performance insights.

---

## References

- **Ising Model**: [L. Onsager, Phys. Rev. 65, 117 (1944)](https://doi.org/10.1103/PhysRev.65.117)  
- **Metropolis Algorithm**: [N. Metropolis et al., J. Chem. Phys. 21, 1087 (1953)](https://doi.org/10.1063/1.1699114)  
- **PyTorch Documentation**: [https://pytorch.org](https://pytorch.org)

---

This project serves as a minimal example of combining **statistical physics** and **deep learning**. Feel free to fork the code, experiment with larger lattices, vary the Monte Carlo sweeps, or integrate other physics-informed sampling methods. Contributions or comments are always welcome!

