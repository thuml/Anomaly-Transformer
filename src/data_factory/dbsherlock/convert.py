import argparse
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import scipy

from src.data_factory.dbsherlock.data import AnomalyDataset
from src.data_factory.dbsherlock.utils import process_dataset

logger = logging.getLogger("DataConverter")


def main(input_path: str, output_dir: str, prefix: str) -> None:
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load .mat file (i.e., original dataset from DBSherlock author's repository)
    original_data = scipy.io.loadmat(input_path)

    # Create anomaly dataset
    logger.info(f"Create anomaly dataset from {input_path}")
    test_dataset: AnomalyDataset = process_dataset(
        causes_as_np=original_data["causes"],
        dataset_as_np=original_data["test_datasets"],
        normal_regions=original_data["normal_regions"],
        abnormal_regions=original_data["abnormal_regions"],
    )

    # Write to file
    output_file_path = os.path.join(output_dir, f"{prefix}_test.json")
    logger.info(f"Writing {len(test_dataset)} data to {output_file_path}")
    file_utils.write_json_file(test_dataset.dic, output_file_path)

    # Create compound anomaly dataset if exists
    if "compound_datasets" in original_data:
        logger.info(f"Create compound anomaly dataset from {input_path}")
        compound_dataset: AnomalyDataset = process_dataset(
            causes_as_np=original_data["compound_causes"],
            dataset_as_np=original_data["compound_datasets"],
            normal_regions=original_data["normal_regions_compound"],
            abnormal_regions=original_data["abnormal_regions_compound"],
        )
        # Write to file
        output_file_path = os.path.join(output_dir, f"{prefix}_test_compound.json")
        logger.info(f"Writing {len(compound_dataset)} data to {output_file_path}")
        file_utils.write_json_file(compound_dataset.dic, output_file_path)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Convert .mat to .csv")
    parser.add_argument(
        "--input",
        type=str,
        default="data/original_dataset/dbsherlock_dataset_tpcc_16w.mat",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/converted_dataset",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        choices=["tpcc_16w", "tpcc_500w", "tpce_3000"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    # Arguments
    input_path = args.input
    output_dir = args.out_dir
    prefix = args.prefix

    main(input_path=input_path, output_dir=output_dir, prefix=prefix)
    logger.info("Done!")
