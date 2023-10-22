import argparse
import json
import os
import pickle
from typing import *
import random
import numpy as np
import tqdm
from src.data_factory.dbsherlock.utils import anomaly_causes

def partition(train_num, seed=None):
    if seed is not None:
        random.seed(seed)
    
    dataset_ids = list(range(11))
    random.shuffle(dataset_ids)
    
    cut1 = train_num
    cut2 = cut1 + 1

    train_ids = dataset_ids[:cut1]
    val_ids = dataset_ids[cut1:cut2]
    test_ids = dataset_ids[cut2:]
    
    return train_ids, val_ids, test_ids

def create_data(
    values: np.array, abnormal_regions: List[int]
) -> Tuple[List[int], np.array]:
    """values: (time step, attribute num)"""
    label = [int(idx in abnormal_regions) for idx in range(values.shape[0])]
    filtered_values = values[:, 2:]
    return label, filtered_values


def process_data(data: Dict, train_num: int = 8, seed: int = 0) -> Tuple[Dict, int, int, int]:
    per_cause_cnt: Dict[str, int] = {}
    total_num = 0
    anomaly_num = 0
    train_values = []
    train_labels = []
    train_classes = []
    val_values = []
    val_labels = []
    val_classes = []
    test_values = []
    test_labels = []
    test_classes = []
    distinct_time_range = set()
    train_ids, val_ids, test_ids = partition(train_num, seed)
    for data_dic in tqdm.tqdm(data["data"]):
        cause = data_dic["cause"]
        values = data_dic["values"]  # Shape: (time step, attribute num)
        values_np = np.array(values)
        abnormal_regions = data_dic["abnormal_regions"]
        distinct_time_range.add(values_np.shape[0])

        # Pass the data with poor physical design (Following the MacroBase paper)
        if cause == "Poor Physical Design":
            continue

        # Count total number of data
        total_num += len(values)
        anomaly_num += len(abnormal_regions)

        # Count number of data for each cause
        if cause in per_cause_cnt:
            per_cause_cnt[cause] += 1
        else:
            per_cause_cnt[cause] = 0

        # Create data
        labels, values = create_data(values_np, abnormal_regions)
        
        # Split dataset
        if per_cause_cnt[cause] in train_ids:
            # Add to training set
            train_labels.append(labels)
            train_values.append(values)
            train_classes.append([anomaly_causes.index(cause)] * len(labels))
        elif per_cause_cnt[cause] in val_ids:
            # Add to validation set
            val_labels.append(labels)
            val_values.append(values)
            val_classes.append([anomaly_causes.index(cause)] * len(labels))
        else:
            # Add to testing set
            test_labels.append(labels)
            test_values.append(values)
            test_classes.append([anomaly_causes.index(cause)] * len(labels))

    processed_data = {
        "train": train_values,
        "train_label": train_labels,
        "train_class": train_classes,
        "val": val_values,
        "val_label": val_labels,
        "val_class": val_classes,
        "test": test_values,
        "test_label": test_labels,
        "test_class": test_classes,
    }

    return processed_data, total_num, anomaly_num


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="/root/Anomaly_Explanation/dataset/dbsherlock/converted/tpcc_500w_test.json",
        help="path of the dbsherlock dataset",
    )
    parser.add_argument(
        "--output_path",
        default="/root/Anomaly_Explanation/dataset/dbsherlock/processed/tpcc_500w/",
        help="path of the processed dataset",
    )
    parser.add_argument(
        "--random_seed", 
        default=1,
        help="random seed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(input_path, "r") as file:
        data = json.load(file)

    processed_data, total_num, anomaly_num = process_data(data=data, seed=args.random_seed)
    print(f"Total:{total_num}")
    print(f"Anomaly:{anomaly_num}")
    print(f"AR:{float(anomaly_num/total_num)}")
    for mode in [
        "train",
        "train_label",
        "train_class",
        "val",
        "val_label",
        "val_class",
        "test",
        "test_label",
        "test_class",
    ]:
        save_path = output_path + "/" + mode + ".pkl"
        with open(save_path, "wb") as file:
            pickle.dump(processed_data[mode], file)
        print(f"{mode}_num:{len(processed_data[mode])}")
