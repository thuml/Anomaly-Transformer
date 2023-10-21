import logging
from typing import *

import hkkang_utils.list as list_utils
import numpy as np
import tqdm

from src.data_factory.dbsherlock.data import AnomalyData, AnomalyDataset

logger = logging.getLogger("DBSherlockDataConverter")

anomaly_causes = [
    "Poorly Written Query",
    "Poor Physical Design",
    "Workload Spike",
    "I/O Saturation",
    "DB Backup",
    "Table Restore",
    "CPU Saturation",
    "Flush Log/Table",
    "Network Congestion",
    "Lock Contention",
]


def to_zero_based_index(indices: List[int]) -> List[int]:
    assert all([i > 0 for i in indices]), f"Indices are already zero-based: {indices}"
    return [i - 1 for i in indices]


def np_region_to_list(region: np.ndarray) -> List[int]:
    if len(region):
        assert len(region) == 1, f"region shape: {region.shape}"
        return [d.item() for d in region[0]]
    return []


def np_dataset_to_test_cases(
    dataset_as_np: np.ndarray,
) -> List[Tuple[List[str], List[np.ndarray]]]:
    test_cases: List[Tuple[List[str], List[np.ndarray]]] = []
    # Parse different test cases
    for idx, test_case in enumerate(tqdm.tqdm(dataset_as_np), start=1):
        # Parse for different duration or start time of the anomaly
        attributes_list: List[str] = []
        values_list: List[np.ndarray] = []
        for variation_idx, data in enumerate(test_case):
            # Decouple numpy array
            item = data[0][0]
            # Get values
            values = item[0]
            # Get attributes
            attributes = [d.item() for d in item[1][0]]
            # logger.info(
            #     f"Test case {idx}-{variation_idx}. attribute num: {len(attributes)}, value shape: {values.shape}"
            # )
            attributes_list.append(attributes)
            values_list.append(values)
        test_cases.append((attributes_list, values_list))

    return test_cases


def process_dataset(
    causes_as_np: np.ndarray,
    dataset_as_np: np.ndarray,
    normal_regions: np.ndarray,
    abnormal_regions: np.ndarray,
) -> AnomalyDataset:
    causes = [d.item() for d in causes_as_np[0]]
    data_list: List[AnomalyData] = []
    # Convert dataset to test cases
    test_cases = np_dataset_to_test_cases(dataset_as_np)
    # Create AnomalyData for each test case (i.e., for each anomaly causes)
    for cause, test_case, n_regs, abn_regs in list_utils.safe_zip(
        causes, test_cases, normal_regions, abnormal_regions
    ):
        attributes_list, values_list = test_case
        # There are 11 data for each test case
        for attributes, values, n_reg, abn_reg in list_utils.safe_zip(
            attributes_list, values_list, n_regs, abn_regs
        ):
            data_list.append(
                AnomalyData(
                    cause=cause,
                    attributes=attributes,
                    values=values.tolist(),
                    normal_regions=to_zero_based_index(np_region_to_list(n_reg)),
                    abnormal_regions=to_zero_based_index(np_region_to_list(abn_reg)),
                )
            )

    # Extract anomaly causes
    anomaly_causes: List[str] = []
    for data in data_list:
        anomaly_cause = data.cause
        if anomaly_cause not in anomaly_causes:
            anomaly_causes.append(anomaly_cause)

    return AnomalyDataset(
        causes=anomaly_causes,
        data=data_list,
    )
