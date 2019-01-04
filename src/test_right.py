

def convert_inf(self ,start ,end):
    if end <= start:
        print("warning: Please correct your input index")
        return
    # change Inf of GO to Max value of GO
    inf_ind = np.where(self.all_data[:, start:end] == float("inf"))
    self.all_data[:, start:end][inf_ind] = -1
    max_GO = np.max(self.all_data[:, start:end])
    self.all_data[:, start:end][inf_ind] = max_GO


def calculate_mean_std(self):
    # use all_list to get mean and std
    mean_data = np.mean(self.all_data, axis=0)
    std_data = np.std(self.all_data, axis=0)
    return torch.tensor(mean_data, dtype=torch.float), \
           torch.tensor(std_data, dtype=torch.float)

def transform(self, raw_data):
    # self.mean_use = np.delete(self.mean, self.del_col, 0)
    # self.std_use = np.delete(self.std, self.del_col, 0)
    # return (raw_data - self.mean_use)/self.std_use
    return (raw_data - self.mean) / self.std
    # return raw_data




path = "/Users/zhangyiding/Documents/yd_singlecell/py_cipher/data/test.txt"
all_unlabeled_list = []
with open(path, "r") as f:
    content = f.read().split("\n")
    for l in content:
        all_unlabeled_list.append([float(i) for i in l.split("\t")[2:]])

self.mean, self.std = self.calculate_mean_std()
