import argparse
import json
import numpy as np
import os

def process_data(data):
    cause_count = {}
    cnt=0
    cnt_val=0
    cnt_test=0
    total_num = 0
    anomaly_num = 0
    for i, data_dict in enumerate(data):
        if data_dict['cause']=='Poor Physical Design':
            continue
        total_num+=len(data_dict['values'])
        anomaly_num+=len(data_dict['abnormal_regions'])
        
        if data_dict['cause'] not in list(cause_count.keys()):
            cause_count[data_dict['cause']]=0
        else:
            cause_count[data_dict['cause']]+=1
        
        if cause_count[data_dict['cause']]<8:
            if cnt == 0:
                train_array =  np.array(data_dict["values"])[:, 2:]
                cnt=1
            else:
                train_array = np.concatenate((train_array, np.array(data_dict["values"])[:, 2:]), axis=0)
        elif cause_count[data_dict['cause']]==8:
            if cnt_val == 0:
                val_array =  np.array(data_dict["values"])[:, 2:]
                cnt_val=1
            else:
                val_array = np.concatenate((val_array, np.array(data_dict["values"])[:, 2:]), axis=0)
        else:
            test_label_list = []
            for idx in range(len(data_dict["values"])):
                if idx in data_dict["abnormal_regions"]:
                    test_label_list.append(1)
                else:
                    test_label_list.append(0)
            if cnt_test == 0:
                test_array = np.array(data_dict["values"])[:, 2:]
                test_label = np.array(test_label_list)
                cnt_test=1
            else:
                test_array = np.concatenate((test_array, np.array(data_dict["values"])[:, 2:]), axis=0)
                test_label = np.concatenate((test_label, np.array(test_label_list)), axis=0)
    processed_data = {"train": train_array, "val": val_array, "test": test_array, "test_label": test_label}
    return processed_data, total_num, anomaly_num

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default='/home/shpark/Anomaly_Explanation/dataset/converted_dataset/tpcc_500w_test.json', help="path of the dbsherlock dataset")
    parser.add_argument("--output_path", default='/home/shpark/Anomaly_Explanation/dataset/DBS_C_new', help="path of the processed dataset")
    return parser.parse_args()

if __name__=="__main__":
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
    for mode in ["train", "val", "test", "test_label"]:
        save_path = output_path + "/" + mode + ".npy"
        np.save(save_path, processed_data[mode])
        print(f"{mode}_num:{len(processed_data[mode])}")