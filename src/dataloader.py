import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Cipher_Dataloader(Dataset):
    def __init__(self, data_path, data_type, train_ind):
        self.data_path = data_path
        self.data_type = data_type

        # read from file
        self.positive_input_list, self.positive_label_list, \
        self.negative_input_list, self.negative_label_list = self.read_all_data(data_path)
        # print(self.positive_input_list)

        # combine data and convert to np.array
        self.all_input_list = self.positive_input_list + self.negative_input_list
        self.all_input_array = np.array(self.all_input_list)
        self.positive_input_list = np.array(self.positive_input_list, dtype=np.float32)
        self.negative_input_list = np.array(self.negative_input_list, dtype=np.float32)
        #change inf of GO to max value of GO
        self.convert_inf(5,22)
        # change inf of distance to max value of distance
        self.convert_inf(4,5)

        # calculate mean and std for all input
        self.mean, self.std = self.calculate_mean_std()

        # separate train and test data
        pos_ind, neg_ind = train_ind
        if self.data_type == "train":
            self.pos_data_list = [(torch.tensor(self.positive_input_list[ind]),
                                   torch.tensor(self.positive_label_list[ind]))
                                  for ind in pos_ind]
            self.neg_data_list = [(torch.tensor(self.negative_input_list[ind]),
                                   torch.tensor(self.negative_label_list[ind]))
                                  for ind in neg_ind]
        elif self.data_type == "test":
            self.data_list = [(torch.tensor(self.positive_input_list[ind]),
                               torch.tensor(self.positive_label_list[ind]))
                              for ind in pos_ind]
            for ind in neg_ind:
                self.data_list.append((torch.tensor(self.negative_input_list[ind]),
                                       torch.tensor(self.negative_label_list[ind])))

    def read_all_data(self, data_path):
        positive_file = os.path.join(data_path, "input_pos_add_distance.txt")
        negative_file = os.path.join(data_path, "input_neg_add_distance.txt")
        # positive list
        positive_input_list = []
        positive_label_list = []
        with open(positive_file, "r") as f:
            content = f.read().split("\n")
            for l in content:
                positive_input_list.append([float(i) for i in l.split("\t")[2:]])
                positive_label_list.append([1.0, 0.0])
        # negative list
        negative_input_list = []
        negative_label_list = []
        with open(negative_file, "r") as f:
            content = f.read().split("\n")
            for l in content:
                negative_input_list.append([float(i) for i in l.split("\t")[2:]])
                negative_label_list.append([0.0, 1.0])
        # print(len(positive_input_list))
        # print(len(negative_input_list))
        # print("pos_raw_input:")
        # print(positive_input_list)
        return positive_input_list, positive_label_list,\
               negative_input_list, negative_label_list

    def convert_inf(self,start,end):
        #convert self.all_input_array, self.positive_input_list and self.negative_input_list to np.array before use this function
        if end <= start:
            print("warning: Please correct your input index")
            return
        # change Inf of GO to Max value of GO
        inf_ind = np.where(self.all_input_array[:, start:end] == float("inf"))
        self.all_input_array[:, start:end][inf_ind] = -1
        max_GO = np.max(self.all_input_array[:, start:end])
        self.all_input_array[:, start:end][inf_ind] = max_GO
        # change Inf of input list
        inf_ind = np.where(self.positive_input_list[:, start:end] == float("inf"))
        self.positive_input_list[:, start:end][inf_ind] = max_GO
        inf_ind = np.where(self.negative_input_list[:, start:end] == float("inf"))
        self.negative_input_list[:, start:end][inf_ind] = max_GO

    def calculate_mean_std(self):
        mean_data = np.mean(self.all_input_array, axis=0)
        std_data = np.std(self.all_input_array, axis=0)
        return torch.tensor(mean_data, dtype=torch.float), \
               torch.tensor(std_data, dtype=torch.float)

    def transform(self, raw_data):
        # return (raw_data - self.mean)/self.std
        return raw_data

    def __len__(self):
        if self.data_type == "train":
            return len(self.pos_data_list)*len(self.neg_data_list)
        if self.data_type == "test":
            return len(self.data_list)

    def __getraw__(self, index):
        if self.data_type == "train":
            pos_in, pos_out = self.pos_data_list[index % len(self.pos_data_list)]
            neg_in, neg_out = self.neg_data_list[index % len(self.neg_data_list)]
            return pos_in, pos_out, neg_in, neg_out
        if self.data_type == "test":
            raw_int, raw_out = self.data_list[index]
            return raw_int, raw_out

    def __getitem__(self, index):
        if self.data_type == "train":
            pos_in, pos_out, neg_in, neg_out = self.__getraw__(index)
            normalized_pos_in = self.transform(pos_in)
            normalized_neg_in = self.transform(neg_in)
            return normalized_pos_in, pos_out, normalized_neg_in, neg_out
        elif self.data_type == "test":
            raw_int, raw_out = self.__getraw__(index)
            normalized_in = self.transform(raw_int)
            return normalized_in, raw_out


if __name__ == '__main__':
    for i in range(1):
        pos_ind = [ind for ind in range(15)]
        neg_ind = [ind for ind in range(80)]
        train_ind = [pos_ind, neg_ind]
        train_dataset = Cipher_Dataloader("../data/", "train", train_ind)
        train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
        for pos_input, pos_label, neg_input, neg_label in train_loader:
             print(pos_input.size())
             print(pos_label.size())
             print(neg_input.size())
             print(neg_label.size())
             print(neg_input)
             break
