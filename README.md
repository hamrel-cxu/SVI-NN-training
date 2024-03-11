# SVI NN training

Official implementation for [An alternative approach to train neural networks using monotone variational inequality](https://arxiv.org/abs/2202.08876).

Please direct all implementation-related inquiries to cxu310@gatech.edu.

The released code compares SVI vs. SGD on the large OGB node classification task
- Training: [SVI_OGB_128width](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/SVI_OGB_128width.ipynb), [SVI_OGB_256width](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/SVI_OGB_256width.ipynb), [SVI_OGB_512width](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/SVI_OGB_512width.ipynb)
- Evaluation: [SVI_OGB_results](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/SVI_OGB_results.ipynb)

Citation:
```
@article{xu2022alternative,
  title={An alternative approach to train neural networks using monotone variational inequality},
  author={Xu, Chen and Cheng, Xiuyuan and Xie, Yao},
  journal={arXiv preprint arXiv:2202.08876},
  year={2022}
}
```
<!-- ## Table of Contents
* [Full results](#full-results)
 -->

<!-- ## Full results -->
<!-- - Please see [real_data_OGB.ipynb](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/real_data_OGB.ipynb) for applying our `SVI` technique on one of the realistic real-data [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) node prediction task from the [Open Graph Benchmark](https://ogb.stanford.edu/). We realize that the original training data designed by OGB *peaks* into the test data (due to undirected edges corresponding to citation between papers), so that adjustments are made to only include citation among papers in the training data. The overall accuracies may thus be less than those on the leaderboard but are still comparable over different optimization techniques under the same neural network architecture.


- Codes on generating other results will be released upon publication. The current paper is under review by NeurIPS 2022.
- The illustration below demonstrates the performance of `SVI` (trained under SGD or Adam) against SGD or Adam on the one of the representative dataset on OGB; results are reproducible upon executing the Jupyter notebook above. In particular, the top figure shows `SVI` always improves the initial training phases by converging faster to desired accuracies. The bottom figures shows that after training until convergence, `SVI` yields competitive performance as gradient-based methods.

Accuracies during initial training stages         |
:-------------------------:
![](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/OGB_initial_epochs.png)
**Accuracies over all epochs**          |
![](https://github.com/hamrel-cxu/SVI-NN-training/blob/main/OGB_all_epochs.png) -->



