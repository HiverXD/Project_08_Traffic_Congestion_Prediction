# Traffic Congestion Prediction with Lightweight STGNN Models and Synthetic Datasets

This repository contains the code for a research project focused on developing lightweight and efficient models for traffic congestion prediction, leveraging novel synthetic datasets and advanced spatio-temporal modeling techniques.

---

## 🎯Project Overview

The increasing demand for smart city solutions and autonomous driving systems highlights the critical need for accurate traffic volume prediction. However, existing challenges include the scarcity of lightweight synthetic datasets and the difficulty of simultaneously achieving model lightweighting and robust performance in spatio-temporal graph data.

This project addresses these challenges by:
1.  **Developing a Lightweight Traffic Volume Simulation and Synthetic Dataset Generation**: Creating a scalable and flexible synthetic dataset that realistically mimics urban traffic patterns, including rush hours and daily/weekly variations.
2.  **Proposing a Lightweight Spatio-Temporal Traffic Prediction Model (STLinear with Attention Bias)**: Integrating state-of-the-art time series prediction concepts (inspired by T-Graphormer and Exphormer) with a novel attention mechanism to enhance efficiency and prediction accuracy.
3.  **Implementing and Evaluating Models**: Providing a robust framework for training, evaluating, and optimizing various spatio-temporal models, including baselines and the proposed STLinear variants.

---

## 🏅Key Features

* **Synthetic Traffic Dataset Generation**:
    * A lightweight traffic simulation that models real-world urban traffic.
    * Generates spatio-temporal traffic volume data represented as graphs (intersections/points as nodes, roads as edges).
    * Supports flexible additions and extensions of traffic patterns (e.g., rush hour, daily/weekly variations).
    * Data is stored in `numpy` format for easy modification, loading, and visualization.

![dataset_gif](/Project_08_Traffic_Congestion_Prediction/figures/dataset.gif)

* **STLinear Model with Structural Attention Bias**:
    * Combines a Linear-based temporal processing module with **structural Attention Bias (Hop-biased and SPE)**.
    * **Hop-biased bias** accelerates information propagation between adjacent nodes.
    * **SPE (Structural Position Embedding) bias** facilitates rapid learning of global relationships between functionally similar nodes.
    * Achieves high computational efficiency and faster convergence by reducing parameter count while incorporating meaningful spatial structures.

![benchmark_test](/Project_08_Traffic_Congestion_Prediction/figures/benchmark_test.png)

* **Faster Training Convergence**:
    * The explicit incorporation of graph structural information as an inductive bias into the attention mechanism enables the model to update parameters more accurately from the early stages of training.
    * This leads to improved convergence speed and better initial performance compared to traditional baselines.
* **Comprehensive Model Evaluation**:
    * Includes a `Trainer` class and performance index calculation utilities (`calculate_performance_index.py`) for systematic model training and evaluation.
    * Supports hyperparameter tuning (`hyperparameter_tuning.ipynb`) to optimize model performance.

![edge_prediction](/Project_08_Traffic_Congestion_Prediction/figures/edge_04_prediction.png)

---

## Repository Structure

The repository is organized as follows:

```
.
├── dataset_generation/               # Scripts and notebooks for synthetic traffic dataset generation
│   ├── 00_random_city_generation.ipynb   # Generate random city layouts for simulation
│   ├── 01_graph_preprocessing.ipynb      # Preprocessing graph data
│   ├── 02_vehicle_environment_test.ipynb # Test vehicle movement within the simulated environment
│   └── ...                               # Other dataset generation/analysis notebooks
├── src/                              # Main source code for models, training, and utilities
│   ├── dataset/                      # Dataset handling and configuration
│   │   ├── dataset_config.py
│   │   └── traffic_dataset.py
│   ├── models/                       # Implementations of various spatio-temporal prediction models
│   │   ├── STLinear.py
│   │   ├── STLinear_biased_models.py # Derived STLinear model (proposed)
│   │   └── ...                       # Other baseline and custom models
│   ├── utils/                        # Utility functions for training, evaluation, and visualization
│   │   ├── Trainer.py
│   │   ├── calculate_performance_index.py
│   │   └── visualization.py
│   └── ...                           # Jupyter notebooks for training, analysis, and tuning
├── .gitignore                        # Specifies intentionally untracked files to ignore
├── README.md                         # Project overview and instructions
├── LICENSE
└── requirements.txt                  # Python dependencies
```

---

## Getting Started

### Prerequisites

* Python 3.9
* PyTorch
* Other dependencies listed in `requirements.txt`


### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/HiverXD/Project_08_Traffic_Congestion_Prediction.git](https://github.com/HiverXD/Project_08_Traffic_Congestion_Prediction.git)
    cd Project_08_Traffic_Congestion_Prediction
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Generate Synthetic Data**:
    * Run the notebooks in `dataset_generation/` to create your own synthetic traffic datasets. Start with `00_random_city_generation.ipynb` and `02_vehicle_environment_test.ipynb`.
2.  **Train Models**:
    * Explore the `src/` directory. `src/baseline_train.ipynb` and `src/baseline_train_refactoring.ipynb` provide examples of training pipelines.
    * Experiment with different models in `src/models/`, including the proposed `STLinear_deriven.py`.
3.  **Evaluate and Analyze**:
    * Use `src/utils/Trainer.py` and `src/utils/calculate_performance_index.py` for model evaluation.
    * Refer to `src/main.ipynb` for analysis examples.

## ⚖️License

MIT License