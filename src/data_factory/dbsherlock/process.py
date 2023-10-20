import argparse
import json
import os
from typing import *

import numpy as np
import tqdm


def create_data(
    values: np.array, abnormal_regions: List[int]
) -> Tuple[List[int], np.array]:
    """values: (time step, attribute num)"""
    label = [int(idx in abnormal_regions) for idx in range(values.shape[0])]
    filtered_values = values[:, 2:]
    return label, filtered_values


def process_data(data: Dict, train_num: int = 8) -> Tuple[Dict, int, int]:
    per_cause_cnt: Dict[str, int] = {}
    total_num = 0
    anomaly_num = 0
    train_values = []
    train_labels = []
    val_values = []
    val_labels = []
    test_values = []
    test_labels = []
    for data_dic in tqdm.tqdm(data["data"]):
        cause = data_dic["cause"]
        values = data_dic["values"]  # Shape: (time step, attribute num)
        values_np = np.array(values)
        abnormal_regions = data_dic["abnormal_regions"]

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
            per_cause_cnt[cause] = 1

        # Split dataset
        labels, values = create_data(values_np, abnormal_regions)
        if per_cause_cnt[cause] < train_num:
            # Create training set
            train_labels.append(labels)
            train_values.append(values)
        elif per_cause_cnt[cause] == train_num:
            # Create validation set
            val_labels.append(labels)
            val_values.append(values)
        else:
            # Create testing set
            test_labels.append(labels)
            test_values.append(values)

    processed_data = {
        "train": np.vstack(train_values),
        "train_label": sum(train_labels, []),
        "val": np.vstack(val_values),
        "val_label": sum(val_labels, []),
        "test": np.vstack(test_values),
        "test_label": sum(test_labels, []),
    }
    return processed_data, total_num, anomaly_num


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="/root/Anomaly_Explanation/dataset/converted_dataset/tpcc_500w_test.json",
        help="path of the dbsherlock dataset",
    )
    parser.add_argument(
        "--output_path",
        default="/root/Anomaly_Explanation/dataset/processed_dataset",
        help="path of the processed dataset",
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

    processed_data, total_num, anomaly_num = process_data(data)
    print(f"Total:{total_num}")
    print(f"Anomaly:{anomaly_num}")
    print(f"AR:{float(anomaly_num/total_num)}")
    for mode in ["train", "train_label", "val", "val_label", "test", "test_label"]:
        save_path = output_path + "/" + mode + ".npy"
        np.save(save_path, processed_data[mode])
        print(f"{mode}_num:{len(processed_data[mode])}")
