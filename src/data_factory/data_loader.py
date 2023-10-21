import collections
import json
import math
import numbers
import os
import pickle
import random
from typing import *

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from src.data_factory.dbsherlock.utils import anomaly_causes

random.seed(0)


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + "/train.csv")
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + "/test.csv")

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + "/test_label.csv").values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class DBSSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", cause="all"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.cause = cause
        self.scaler = StandardScaler()
        self.__post__init__(data_path)

    def __post__init__(self, data_path: str) -> None:
        # Load training data
        data = self._load_pickle(os.path.join(data_path, "train.pkl"))
        self.scaler.fit(np.vstack(data))
        data = [self.scaler.transform(datum) for datum in data]
        labels = self._load_pickle(os.path.join(data_path, "train_label.pkl"))
        classes = self._load_pickle(os.path.join(data_path, "train_class.pkl"))
        data, labels, classes = self._filter_data(data, labels, classes, self.cause)
        self.train_data = self._preprocess_data(data, kernel_size=self.step)
        self.train_labels = self._preprocess_data(labels, kernel_size=self.step)
        self.train_classes = self._preprocess_data(classes, kernel_size=self.step)
        # Load validation data
        data = self._load_pickle(os.path.join(data_path, "val.pkl"))
        data = [self.scaler.transform(datum) for datum in data]
        labels = self._load_pickle(os.path.join(data_path, "val_label.pkl"))
        classes = self._load_pickle(os.path.join(data_path, "val_class.pkl"))
        data, labels, classes = self._filter_data(data, labels, classes, self.cause)
        self.val_data = self._preprocess_data(data, kernel_size=self.step)
        self.val_labels = self._preprocess_data(labels, kernel_size=self.step)
        self.val_classes = self._preprocess_data(classes, kernel_size=self.step)

        # Load test data
        data = self._load_pickle(os.path.join(data_path, "test.pkl"))
        data = [self.scaler.transform(datum) for datum in data]
        labels = self._load_pickle(os.path.join(data_path, "test_label.pkl"))
        classes = self._load_pickle(os.path.join(data_path, "test_class.pkl"))
        data, labels, classes = self._filter_data(data, labels, classes, self.cause)
        self.test_data = self._preprocess_data(data, kernel_size=self.step)
        self.test_labels = self._preprocess_data(labels, kernel_size=self.step)
        self.test_classes = self._preprocess_data(classes, kernel_size=self.step)

        # Load test data for thresholding
        data = self._load_pickle(os.path.join(data_path, "test.pkl"))
        data = [self.scaler.transform(datum) for datum in data]
        labels = self._load_pickle(os.path.join(data_path, "test_label.pkl"))
        classes = self._load_pickle(os.path.join(data_path, "test_class.pkl"))
        data, labels, classes = self._filter_data(data, labels, classes, self.cause)
        self.test_data_threshold = self._preprocess_data(
            data, kernel_size=self.win_size
        )
        self.test_labels_threshold = self._preprocess_data(
            labels, kernel_size=self.win_size
        )
        self.test_classes_threshold = self._preprocess_data(
            classes, kernel_size=self.win_size
        )

    def _filter_data(
        self, data: List[Any], labels, classes, cause: str
    ) -> Tuple[List, List, List]:
        if self.cause == "all":
            return data, labels, classes
        else:
            # Filter
            cause_idx = anomaly_causes.index(self.cause)
            data = [
                datum for datum, class_ in zip(data, classes) if class_[0] == cause_idx
            ]
            labels = [
                label
                for class_, label in zip(classes, labels)
                if class_[0] == cause_idx
            ]
            classes = [class_ for class_ in classes if class_[0] == cause_idx]
            return data, labels, classes

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.train_data)

        elif self.mode == "val":
            return len(self.val_data)
        elif self.mode == "test":
            return len(self.test_data)
        else:
            return len(self.test_data_threshold)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == "train":
            return np.float32(self.train_data[index]), np.float32(
                self.train_labels[index]
            )
        elif self.mode == "val":
            return np.float32(self.val_data[index]), np.float32(self.val_labels[index])
        elif self.mode == "test":
            return np.float32(self.test_data[index]), np.float32(
                self.test_labels[index]
            )
        else:
            return np.float32(self.test_data_threshold[index]), np.float32(
                self.test_labels_threshold[index]
            )

    def _load_pickle(self, path: str) -> List[Union[np.ndarray, int]]:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def _preprocess_datum(self, datum: List[Any], kernel_size: int) -> List[List[Any]]:
        window_data_list = []
        for i in range(len(datum) // kernel_size):
            window_data_list.append(datum[i * kernel_size : (i + 1) * kernel_size])
        if len(datum) % kernel_size == 0:
            window_data_list.append(datum[-kernel_size:])
        return window_data_list

    def _preprocess_data(
        self, data: List[List[Any]], kernel_size: int
    ) -> List[List[Any]]:
        all_data = []
        for datum in data:
            datum = self._preprocess_datum(datum, kernel_size)
            all_data.extend(datum)
        return all_data


def get_loader_segment(
    data_path,
    batch_size,
    win_size=100,
    step=100,
    mode="train",
    dataset="KDD",
    cause="all",
):
    if dataset == "SMD":
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == "MSL":
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif dataset == "SMAP":
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif dataset == "PSM":
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif dataset == "DBS":
        dataset = DBSSegLoader(data_path, win_size, step, mode, cause)
    shuffle = False
    if mode == "train":
        shuffle = True

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    return data_loader
