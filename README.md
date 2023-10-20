# Anomaly-Transformer (ICLR 2022 Spotlight)
Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

Unsupervised detection of anomaly points in time series is a challenging problem, which requires the model to learn informative representation and derive a distinguishable criterion. In this paper, we propose the Anomaly Transformer in these three folds:

- An inherent distinguishable criterion as **Association Discrepancy** for detection.
- A new **Anomaly-Attention** mechanism to compute the association discrepancy.
- A **minimax strategy** to amplify the normal-abnormal distinguishability of the association discrepancy.

<p align="center">
<img src=".\pics\structure.png" height = "350" alt="" align=center />
</p>

## Get Started
### Installation
1. Install Python 3.6, PyTorch >= 1.4.0. 
(Thanks Élise for the contribution in solving the environment. See this [issue](https://github.com/thuml/Anomaly-Transformer/issues/11) for details.)
2. Run the following code to install all pip packages:
Run the following code to install all pip packages:
```bash
pip install -r requirements.txt 
```
### Dataset
Download data. You can obtain four benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/9605612594f0423f891e/) or [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.

#### Download
Download SMD, SMAP, PSM, and MSL dataset
```bash
python scripts/download_datasets.py
```

#### Preprocess data

Convert DBSherlock data (.mat file to .json file):
```bash
python src/data_factory/dbsherlock/convert.py \
    --input dataset/dbsherlock/tpcc_16w.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpcc_16w

python src/data_factory/dbsherlock/convert.py \
    --input dataset/dbsherlock/tpcc_500w.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpcc_500w

python src/data_factory/dbsherlock/convert.py \
    --input dataset/dbsherlock/tpce_3000.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpce_3000
```

Convert DBSherlock data to train & validate Anomaly Transformer
```bash
python src/data_factory/dbsherlock/process.py \
    --input_path dataset/dbsherlock/converted/tpcc_16w_test.json \
    --output_path dataset/dbsherlock/processed/tpcc_16w/

python src/data_factory/dbsherlock/process.py \
    --input_path dataset/dbsherlock/converted/tpcc_500w_test.json \
    --output_path dataset/dbsherlock/processed/tpcc_500w/

python src/data_factory/dbsherlock/process.py \
    --input_path dataset/dbsherlock/converted/tpce_3000_test.json \
    --output_path dataset/dbsherlock/processed/tpce_3000/
```
### Train and Evaluate
We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
bash ./scripts/DBS.sh
```
Especially, we use the adjustment operation proposed by [Xu et al, 2018](https://arxiv.org/pdf/1802.03903.pdf) for model evaluation. If you have questions about this, please see this [issue](https://github.com/thuml/Anomaly-Transformer/issues/14) or email us.

## Main Result

We compare our model with 15 baselines, including THOC, InterFusion, etc. **Generally,  Anomaly-Transformer achieves SOTA.**

<p align="center">
<img src=".\pics\result.png" height = "450" alt="" align=center />
</p>

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

## Contact
If you have any question, please contact wuhx23@mails.tsinghua.edu.cn.
