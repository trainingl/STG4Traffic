# STG4Traffic：A Benchmark Study of Using Spatial-Temporal Graph Neural Networks for Traffic Prediction

## 📚 Benchmark Design

**Motivation**：The existing benchmarks for traffic prediction lack a standard and unified benchmark, and the experimental settings are not standardized, with complex environment configurations and poor scalability. Furthermore, there are significant differences between the reported results in the original papers and the results obtained from the existing benchmarks, making it difficult to compare different models based on fair baselines.

Given the heterogeneity of traffic data, which can result in significant variation in model performance across different datasets, we have selected 15 to 20 models with high impact factors on traffic speed (METR-LA, PEMS-BAY) and traffic flow (PEMSD4, PEMSD8), as well as representative models, to construct a benchmark project file for the uniform evaluation of model performance. The selected methods are as follows: 

1. 2018_IJCAI_Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting (STGCN). [Paper](https://arxiv.org/abs/1709.04875)
2. 2018_ICLR_Diffusion convolutional recurrent neural network: Data-driven traffic forecasting (DCRNN). [Paper](https://openreview.net/forum?id=SJiHXGWAZ)
3. 2018_IEEE TITS_T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction (T-GCN). [Paper](https://ieeexplore.ieee.org/abstract/document/8809901)
4. 2019_IJCAI_Graph WaveNet for Deep Spatial-Temporal Graph Modeling (GWNET). [Paper](https://arxiv.org/abs/1906.00121)
5. 2019_AAAI_Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (ASTGCN). [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/3881)
6. 2020_AAAI_Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting (STSGCN). [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5438)
7. 2020_NIPS_Adaptive graph convolutional recurrent network for traffic forecasting (AGCRN). [Paper](https://proceedings.neurips.cc/paper/2020/hash/ce1aad92b939420fc17005e5461e6f48-Abstract.html)
8. 2020_KDD_Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks (MTGNN). [Paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403118)
9. 2020_AAAI_GMAN: A Graph Multi-Attention Network for Traffic Prediction (GMAN). [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5477)
10. 2021_ICLR_Discrete Graph Structure Learning for Forecasting Multiple Time Series (GTS). [Paper](https://arxiv.org/abs/2101.06861)
11. 2021_KDD_Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting (STGODE). [Paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467430)
12. 2021_ACM TKDE_Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution (DGCRN). [Paper](https://dl.acm.org/doi/full/10.1145/3532611)
13. 2021_CIKM_Enhancing the Robustness via Adversarial Learning and Joint Spatial-Temporal Embeddings in Traffic Forecasting (TrendGCN). [Paper](https://arxiv.org/abs/2208.03063)
14. 2022_KDD_MSDR: Multi-Step Dependency Relation Networks for Spatial Temporal Forecasting (GMSDR). [Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539397)
15. 2022_IJCAI_Regularized Graph Structure Learning with Semantic Knowledge for Multi-variates Time-Series Forecasting (RGSL). [Paper](https://arxiv.org/abs/2210.06126)
16. 2022_AAAI_Graph Neural Controlled Differential Equations for Traffic Forecasting (STG-NCDE). [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20587)

**Datasets**：We validate the methods in the benchmark on the following datasets. 

1. METR-LA: Los Angeles Metropolitan Traffic Conditions Data, which records traffic speed data collected at 5-minute intervals by 207 freeway loop detectors from March 2012 to June 2012. 
2. PEMS-BAY: A network representation of 325 traffic sensors in the Bay Area, collected by the California Department of Transportation (CalTrans) Measurement System (PeMS), displaying traffic flow data at 5-minute intervals from January 2017 to May 2017.
3. PEMSD4/8：The traffic flow datasets are collected by the Caltrans Performance Measurement System ([PeMS](http://pems.dot.ca.gov/)) ([Chen et al., 2001](https://trrjournalonline.trb.org/doi/10.3141/1748-12)) in real time every 30 seconds. The traffic data are aggregated into every 5-minute interval from the raw data. 

The detailed statistics of these datasets are as follows:

<center><img src="figure\Datasets.png" style="zoom:80%;text-align: center;" /></center>

[METR-LA & PEMS-BAY]：https://github.com/liyaguang/DCRNN

[PEMSD4 & PEMSD8]：https://github.com/Davidham3/ASTGCN

**Requirements**：The code is built based on Python 3.9.0 and PyTorch 1.8.0.  You can install other dependencies via: 

```bash
pip install -r requirements.txt
```

## 💡 Experimental Performance

To ensure consistency with previous research, we divided the speed data into training, validation, and test sets in a ratio of 7:1:2, while the flow data was divided in a ratio of 6:2:2. If the validation error converges within 15-20 epochs or stops after 100 epochs, the training of the model is stopped and the best model on the validation data is saved. To determine the specific model parameters and settings, including the optimizer, learning rate, loss function, and model parameters, we remained faithful to the original papers while also making multiple tuning efforts to select the best experimental results. In our experiments, we used root mean square error (RMSE), mean absolute error (MAE), and mean absolute percentage error (MAPE) based on masked data as metrics to evaluate the model performance, with zero values being ignored.

<center><img src="figure\speed.png" style="zoom:80%;" /></center>

<center><img src="figure\flow.png" style="zoom:80%;" /></center>

## 🛠️ Usability And Practicality

```bash
cd STG4Traffic/TrafficFlow
```

This directory level enables you to become familiar with the project's public data interface. If you observe carefully, you can see that we have decoupled the data interface, model design, model training and evaluation modules to achieve a structural separation of the benchmarking framework.

| Directory | Explanation                                                  |
| :-------: | :----------------------------------------------------------- |
|   data    | Data files (including historical traffic volume records, connection information of nodes) |
|    lib    | Tool class (data loaders, evaluation metrics, graph construction methods, etc.) |
|   model   | Model design files                                           |
|    log    | The directory where the project logs and models are stored   |

<img src="figure\benchmarkPiple.png" style="zoom:80%;" />

**How can we extend the use of this benchmark?**

Self-Defined Model Design: Create the **ModelName** directory under model and write the **modelname.py** file;

Model Setup, Run and Test: The path STG4Traffic/TrafficFlow/ModelName creates the following 4 files:

- **ModelName_Config.py**: Retrieving the model's parameter configuration entries;

- **ModelName_Utils.py**: Additional tool classes for model setup [optional];

- **ModelName_Trainer.py**: Model trainer, which undertakes the task of training, validation and testing of models throughout the process;

- **ModelName_Main.py**:  Project start-up portal to complete the initialization of model parameters, optimizer, loss function, learning rate decay strategy and other settings;

- **Dataset_ModelName.conf**: Different datasets set different parameter terms for the model [can be multiple].

**Talk is cheap. Show me the code.** You can get a handle on the execution of this benchmark by experimenting with a simple model, such as DCRNN → ModelName, which will help you understand the meaning of the above table of contents.

## 😀 Citation

If you find this repository useful for your work, please consider citing it as follows:


```tex
@article{DBLP:journals/corr/abs-2307-00495,
  author       = {Xunlian Luo and
                  Chunjiang Zhu and
                  Detian Zhang and
                  Qing Li},
  title        = {STG4Traffic: {A} Survey and Benchmark of Spatial-Temporal Graph Neural
                  Networks for Traffic Prediction},
  journal      = {CoRR},
  volume       = {abs/2307.00495},
  year         = {2023}
}
```


[STG4Traffic: A Survey and Benchmark of Spatial-Temporal Graph Neural Networks for Traffic Prediction](https://arxiv.org/abs/2307.00495)

Our research mainly refers to the following works:

[1] AGCRN：https://github.com/LeiBAI/AGCRN

[2] Graph WaveNet：https://github.com/nnzhan/Graph-WaveNet

We hope this research can make positive and beneficial contributions to the field of spatial-temporal prediction!





