import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class Cipher_testDataloader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        #read unlabeled test data and all data
        self.data = self.read_data(data_path)
        #change inf of GO to max value of GO in self.all_data
        #self.convert_inf(4,21)
        # calculate mean and std for self.all_data
        self.mean, self.std = self.calculate_mean_std()
        self.data = torch.tensor(self.data)


    def read_data(self, data_path):
        all_unlabeled_file = os.path.join(data_path, "input_all_log.txt")
        #print(positive_file)
        #all_file = os.path.join(data_path, "input_all_with_labeled_data_log.txt")
        # unlabeled list
        all_unlabeled_list = []
        with open(all_unlabeled_file, "r") as f:
            content = f.read().split("\n")
            for l in content:
                all_unlabeled_list.append([float(i) for i in l.split("\t")[2:]])
        # all_list = []
        # with open(all_file, "r") as f:
        #     content = f.read().split("\n")
        #     for l in content:
        #         all_list.append([float(i) for i in l.split("\t")[2:]])
        all_unlabeled_list = np.array(all_unlabeled_list, dtype=np.float32)
        # all_list = np.array(all_list, dtype=np.float32)
        return all_unlabeled_list

    def convert_inf(self,start,end):
        if end <= start:
            print("warning: Please correct your input index")
            return
        # change Inf of GO to Max value of GO
        inf_ind = np.where(self.all_data[:, start:end] == float("inf"))
        self.all_data[:, start:end][inf_ind] = -1
        max_GO = np.max(self.all_data[:, start:end])
        self.all_data[:, start:end][inf_ind] = max_GO


    def calculate_mean_std(self):
        #use all_list to get mean and std
        mean_data = np.mean(self.data, axis=0)
        std_data = np.std(self.data, axis=0)
        return torch.tensor(mean_data, dtype=torch.float), \
               torch.tensor(std_data, dtype=torch.float)

    def transform(self, raw_data):
        #self.mean_use = np.delete(self.mean, self.del_col, 0)
        #self.std_use = np.delete(self.std, self.del_col, 0)
        #return (raw_data - self.mean_use)/self.std_use
        return (raw_data - self.mean) / self.std
        #return raw_data

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        data_in = self.data[index % len(self.data)]
        normalized_in = self.transform(data_in)
        return normalized_in


if __name__ == '__main__':
    test_dataset = Cipher_testDataloader("../data/")
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    num_iter = 0
    for input in test_loader:
         num_iter += 1
         print(input)
         print(input.size())

         #print(neg_input)
    print(num_iter)

