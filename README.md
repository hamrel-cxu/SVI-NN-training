# SVI NN training
> Implementation and experiments based on the paper "[An alternative approach to train neural networks using monotone variational inequality](https://arxiv.org/abs/2202.08876)". The current paper is under review.
> 
> Please direct all implementation-related inquiries to Chen Xu @ cxu310@gatech.edu.

> Citation:
```
@misc{https://doi.org/10.48550/arxiv.2202.08876,
  doi = {10.48550/ARXIV.2202.08876},
  url = {https://arxiv.org/abs/2202.08876},
  author = {Xu, Chen and Cheng, Xiuyuan and Xie, Yao},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {An alternative approach to train neural networks using monotone variational inequality},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
<!-- ## Table of Contents
* [Full results](#full-results)
 -->

## Full results
- Please see [real_data_OGB.ipynb](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/real_data_OGB.ipynb) for applying our `SVI` technique on one of the realistic real-data [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) node prediction task from the [Open Graph Benchmark](https://ogb.stanford.edu/). We realize that the original training data designed by OGB *peaks* into the test data (due to undirected edges corresponding to citation between papers), so that adjustments are made to only include citation among papers in the training data. The overall accuracies may thus be less than those on the leaderboard but are still comparable over different optimization techniques under the same neural network architecture.


- Codes on generating other results will be released upon publication. The current paper is under review. 

- The illustration below demonstrates the performance of `SVI` (trained under SGD or Adam) against SGD or Adam on the one of the representative dataset on OGB; results are reproducible upon executing the Jupyter notebook above. In particular, the top figure shows `SVI` always improves the initial training phases by converging faster to desired accuracies. The bottom figures shows that after training until convergence, `SVI` yields competitive performance as gradient-based methods.

Accuracies during initial training stages         |
:-------------------------:
![](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/OGB_initial_epochs.png)
**Accuracies over all epochs**          |
![](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/OGB_all_epochs.png)



<!-- - **Required Dependency:** 
  - Basic modules: `numpy, pandas, sklearn, scipy, matplotlib, seaborn, etc.`.
  - Additional modules: `torch` for training fully-connected networks, `torch_geometric` for building graph neural network models, and `networkx` for visualizing graph structures.
- **General Info and Tests:** This work reproduces all experiments in [Training neural networks using monotone variational inequality](https://arxiv.org/abs/2202.08876) (Xu et al. 2022). Note that all except the `utils_gnn_VI.py` files are best executed interactively (e.g., via [Hydrogen](https://atom.io/packages/hydrogen)). In particular, 
  - [simulation_FCNN.py](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/simulation_FCNN.py) contains simulated experiments using fully-connected networks, which appears in Section 5.3.1, 5.3.2 and Appendix B.1, B.2.
  - [simulation_GNN.py](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/simulation_GNN.py) contains simulated experiments using graph neural networks, which appears in Section 5.3.3 and Appendix B.2, B.4.
  - [real_data.py](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/real_data.py) contains real-data experiments using graph neural networks, which appears in Section 5.4 and Appendix B.3.
  - [utils_gnn_VI.py](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/utils_gnn_VI.py) contains all the helper functions. In particular, the function [`train_revised_all_layer`](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/utils_gnn_VI.py#L229) embeds the **SVI** algorithm in training all neural networks examined in this paper -->


