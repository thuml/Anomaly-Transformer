import argparse
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

def get_accuracy(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    pred = np.array(pred)
    gt = np.array(gt)
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    return accuracy, precision, recall, f_score

def feature_extraction_with_pca(data, top_n=1):
    pca = PCA(n_components=top_n)
    pca.fit(data)
    component_weights = pca.components_[0]
    top_indicies = np.argsort(np.abs(component_weights))[-top_n:][::-1].tolist()
    return top_indicies

def normalize(dataset):
    return (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

def load_dataset(input_path):
    dataset = {}
    for partition in ["train", "val", "test"]:
        filename = partition
        data_path = input_path + '/' + filename
        dataset[partition] = normalize(preprocess_data(load_pickle(data_path + ".pkl")))
        filename = partition + "_label"
        data_path = input_path + '/' + filename
        dataset[partition + "_label"] = preprocess_data(load_pickle(data_path + ".pkl"))
    return dataset

def list_subdirectories(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def load_pickle(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def preprocess_data(data):
    all_data = []
    for datum in data:
        all_data.extend(datum)
    return all_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default='/home/shpark/Anomaly_Explanation/dataset/dbsherlock/processed/tpce_3000')
    parser.add_argument("--dim", default=15)
    parser.add_argument("--mode", default='whole', choices=['whole', 'cause'])
    parser.add_argument("--find_best", default=True)
    parser.add_argument("--feature_extraction", default=True)
    return parser.parse_args()

if __name__=='__main__':
    np.random.seed(300)
    args = parse_args()
    mode = args.mode
    dim = args.dim
    input_path = args.input_path
    feature_extraction = args.feature_extraction
    
    if mode == "whole":
        cause_path_list = [input_path]
    else:
        cause_path_list = list_subdirectories(input_path)
        
    for cause_path in cause_path_list:
        print(f"Results of {os.path.basename(cause_path)}")
        dataset = load_dataset(cause_path)
        
        if feature_extraction:
            top_idx = feature_extraction_with_pca(dataset['test'], dim)
            principal_components = dataset['test'][:, top_idx]
        else:
            principal_components = dataset['test']

        try:
            mcd = MinCovDet(support_fraction=1).fit(principal_components)
        except:
            continue
        
        if args.find_best:
            principal_components_train_val = np.concatenate((dataset['train'], dataset['val']), axis=0)[:, top_idx]
            mcd_train_val = MinCovDet(support_fraction=1).fit(principal_components_train_val)
            distances = mcd.mahalanobis(principal_components_train_val-mcd_train_val.location_) ** (.5)
            distances_with_idx = list(zip(range(0, len(distances)), distances))                  
            best_f1 = 0
            best_percentile = None
            for percentile in range(101):
                pctile_cutoff = np.percentile(distances_with_idx, percentile)
                filtered_distances = (distances.tolist() > pctile_cutoff).astype(int)
                accuracy, precision, recall, f_score = get_accuracy(np.concatenate((dataset['train_label'], dataset['val_label']), axis=0).tolist(), filtered_distances.tolist())
                if f_score >= best_f1:
                    best_f1 = f_score
                    best_percentile = percentile
                    best_recall = recall
                    best_precision = precision
            percentile = best_percentile
        else:
            percentile = 99

        distances = mcd.mahalanobis(principal_components-mcd.location_) ** (.5)
        distances_with_idx = list(zip(range(0, len(distances)), distances))        
        pctile_cutoff = np.percentile(distances_with_idx, percentile)
        filtered_distances = (distances.tolist() > pctile_cutoff).astype(int)
        accuracy, precision, recall, f_score = get_accuracy(dataset['test_label'], filtered_distances.tolist())

        print(f"dim: {dim}")
        print(f"percentile for that: {percentile}")
        print(f"recall score: {recall}")
        print(f"precision score: {precision}")
        print(f"f1 score: {f_score}")
        
            