# Anomaly-Transformer
Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy" (ICLR 2022 Spotlight)

Unsupervised detection of anomaly points in time series is a challenging problem, which requires the model to derive a distinguishable criterion. In this paper, we propose the Anomaly Transformer with a new **Anomaly-Attention mechanism** to compute the association discrepancy. A **minimax strategy** is devised to amplify the normal-abnormal distinguishability of the association discrepancy. 



<p align="center">
<img src=".\pics\structure.png" height = "350" alt="" align=center />
</p>


## 

## Get Started

1. Install Python 3.6, PyTorch 1.4.0
2. Download data. You can obtain four benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/) **All the datasets are well pre-processed** .
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
```

## Main Result

We compare our model with 15 baselines, including THOC, InterFusion, etc. Generally,  Anomaly-Transformer achieves SOTA



<p align="center">
<img src=".\pics\result.png" height = "450" alt="" align=center />
</p>


## 

## Citation
If you find this repo useful, please cite our paper. 

```
@inproceedings{
xu2022anomaly,
title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=LzQQ89U1qm_}
}
```

## 

## Contact
If you have any question or want to use the code, please contact xjh20@mails.tsinghua.edu.cn , whx20@mails.tsinghua.edu.cn .
