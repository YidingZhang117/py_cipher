import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class Cipher_Dataloader(Dataset):
    def __init__(self, data_path, data_type, train_ind,test_ind):
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
        self.convert_inf(4,21)
        # change inf of distance to max value of distance
        #self.convert_inf(4,5)
        pos_ind, neg_ind = train_ind
        test_pos_ind, test_neg_ind = test_ind
        del_col = self.find_colind(test_ind)
        self.positive_input_list = np.delete(self.positive_input_list, del_col, 1)
        self.negative_input_list = np.delete(self.negative_input_list, del_col,1)
        # print(del_col)
        self.all_input_array = np.delete(self.all_input_array, del_col,1)
        # print(type(self.positive_input_list))
        # print(type(self.negative_input_list))
        # print(type(self.all_input_array))
        self.all_input_array = self.GOmean(self.all_input_array)
        self.positive_input_list = self.GOmean(self.positive_input_list)
        self.negative_input_list = self.GOmean(self.negative_input_list)

        self.mean, self.std = self.calculate_mean_std()
        #print("after:")
        #print(np.shape(self.positive_input_list))

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
                              for ind in test_pos_ind]
            for ind in test_neg_ind:
                self.data_list.append((torch.tensor(self.negative_input_list[ind]),
                                       torch.tensor(self.negative_label_list[ind])))
    def find_colind(self,test_ind):
        test_pos_ind, test_neg_ind = test_ind
        test_pos_ind_array = np.array(test_pos_ind)
        test_neg_ind_array = np.array(test_neg_ind)
        del_ind_GO = 4 + test_pos_ind_array
        #del_ind_D = 21 + test_pos_ind_array
        #self.del_col = np.append(del_ind_GO,del_ind_D)
        self.del_col = del_ind_GO
        return self.del_col

    def read_all_data(self, data_path):
        positive_file = os.path.join(data_path, "input_positive_log_new.txt")
        #print(positive_file)
        negative_file = os.path.join(data_path, "input_negative_log_new.txt")
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
        #print(end)
        #print(inf_ind)
        self.all_input_array[:, start:end][inf_ind] = -1
        max_GO = np.max(self.all_input_array[:, start:end])
        self.all_input_array[:, start:end][inf_ind] = max_GO
        # change Inf of input list
        inf_ind = np.where(self.positive_input_list[:, start:end] == float("inf"))
        self.positive_input_list[:, start:end][inf_ind] = max_GO
        inf_ind = np.where(self.negative_input_list[:, start:end] == float("inf"))
        self.negative_input_list[:, start:end][inf_ind] = max_GO

    def GOmean(self,data):
        temp = np.mean(data[:,4:20],axis=1)
        temp1 = np.array([temp.tolist()])
        temp2 = data[:,0:4]
        with_mean = np.concatenate((temp2,temp1.T),axis=1)
        # print(np.shape(with_mean))
        return with_mean


        #self.positive_input_list = self.positive_input_list[4:19]
        #self.negative_input_list = self.negative_input_list[4:19]


    def calculate_mean_std(self):
        mean_data = np.mean(self.all_input_array, axis=0)
        std_data = np.std(self.all_input_array, axis=0)
        return torch.tensor(mean_data, dtype=torch.float), \
               torch.tensor(std_data, dtype=torch.float)

    def transform(self, raw_data):
        #self.mean_use = np.delete(self.mean, self.del_col, 0)
        #self.std_use = np.delete(self.std, self.del_col, 0)
        #return (raw_data - self.mean_use)/self.std_use
        raw_data = raw_data.to(dtype = torch.float32)
        return (raw_data - self.mean) / self.std
        #return raw_data

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
    for m in range(1):
        # all_pos_ind = [i for i in range(0, 17)]
        # all_neg_ind = [i for i in range(0, 84)]
        # pos_ind = random.sample(range(0, 17), 15)
        # neg_ind = random.sample(range(0, 84), 80)
        # train_ind = [pos_ind, neg_ind]
        # test_pos_ind = list(set(all_pos_ind).difference(set(pos_ind)))
        # test_neg_ind = list(set(all_neg_ind).difference(set(neg_ind)))
        # test_ind = [test_pos_ind, test_neg_ind]
        pos_ind = [ind for ind in range(15)]
        neg_ind = [ind for ind in range(80)]
        train_ind = [pos_ind, neg_ind]
        test_ind = [[15, 16], [80, 81, 82, 83]]
        train_dataset = Cipher_Dataloader("../data/", "train", train_ind,test_ind)
        train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
        num_iter = 0
        for pos_input, pos_label, neg_input, neg_label in train_loader:
             num_iter += 1
             print(test_ind)
             print(pos_input.size())
             print(pos_label.size())
             print(neg_input.size())
             print(neg_label.size())
             # print(neg_input)
        print(num_iter)

