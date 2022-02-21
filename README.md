# SVI NN training
 Implementation and experiments based on the paper "Training neural networks using monotone variational inequality"

**Note, the code files are currently being polished and documented. A .ipynb demo will be soon provided, as well as a detailed ReadMe file**

## Table of Contents
* [How to use](#how-to-use)
* [Demo and detailed documentation](#demo-and-detailed-documentation)

## How to use
- **Required Dependency:** 
  - Basic modules: `numpy, pandas, sklearn, scipy, matplotlib, seaborn, etc.`.
  - Additional modules: `torch` for training fully-connected networks, `torch_geometric` for building graph neural network models, and `networkx` for visualizing graph structures.
- **General Info and Tests:** This work reproduces all experiments in [Training neural networks using monotone variational inequality](https://arxiv.org/abs/2202.08876) (Xu et al. 2022). Note that all except the `utils_gnn_VI.py` files are best executed interactively (e.g., via [Hydrogen](https://atom.io/packages/hydrogen)). In particular, 
  - [simulation_FCNN.py](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/simulation_FCNN.py) contains simulated experiments using fully-connected networks, which appears in Section 5.3.1, 5.3.2 and Appendix B.1, B.2.
  - [simulation_GNN.py](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/simulation_GNN.py) contains simulated experiments using graph neural networks, which appears in Section 5.3.3 and Appendix B.2, B.4.
  - [real_data.py](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/real_data.py) contains real-data experiments using graph neural networks, which appears in Section 5.4 and Appendix B.3.
  - [utils_gnn_VI.py](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/utils_gnn_VI.py) contains all the helper functions. In particular, the function [`train_revised_all_layer`](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/utils_gnn_VI.py#L229) embeds the **SVI** algorithm in training all neural networks examined in this paper

## Demo and detailed documentation
- To be completed.
