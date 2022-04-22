# Anomaly-Transformer
About Code release for "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy" (ICLR 2022 Spotlight), https://openreview.net/forum?id=LzQQ89U1qm_

## Get Started

1. Install Python 3.6, PyTorch 1.4.0
2. Download data. You can obtain four benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/) **All the datasets are well pre-processed** and can be used easily.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
```

## Main Result

We compare our model with 15 baselines, including THOC, InterFusion, etc. Generally,  Anomaly-Transformer achieves SOTA

## Citation
If you find this repo useful, please cite our paper. 

## Contact
If you have any question or want to use the code, please contact xjh20@mails.tsinghua.edu.cn , whx20@mails.tsinghua.edu.cn .
