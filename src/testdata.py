import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np

from testdata_dataloader import Cipher_testDataloader
from network import Network
from config import DATA_FOLDER #import a variable


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='simple_network', help='model file definition')
    parser.add_argument('-bs',default=4, type=int, help='batch size')
    parser.add_argument('-lt', default=10, type=int, help='Loss file saving refresh interval (seconds)')
    parser.add_argument('-data_path', default='data/', help='Training path')
    parser.add_argument('-rundir', default='../results/test' , help='Running directory')
    parser.add_argument('-ep', default=30, type=int , help='Epochs')
    parser.add_argument('-start_from', default='../results/1231_4+17_normalized_0.014223675709217787/Best_model_period1.pth' , help='Start from previous model')
    #parser.add_argument('-start_from', default='', help='Start from previous model')
    #../results/1125best3_0.019168664837100852/Best_model_period1.pth
    args = parser.parse_args()
    return args


def get_dataloader(args):
    # --- dataloader ---
    full_path = None
    # arguments check
    if args.data_path != '':
        full_path = os.path.join(DATA_FOLDER, args.data_path)
    # error hint
    if full_path is None:
        print("Error: Missing training path for depth!")
        sys.exit(1)
    test_data = Cipher_testDataloader(full_path)
    return test_data



# main
if __name__ == '__main__':
    # --- arguments ---
    args = parseArgs()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # --- dataloader ---
    try:
        test_data = get_dataloader(args)
    except Exception as e:
        print("Error when get dataloader")
        sys.exit(1)

    # run path
    if not os.path.exists(args.rundir):
        os.mkdir(args.rundir)
    torch.save(args, args.rundir+'/args.pth')


    # --- Model and criterion ---
    # model
    config = {}
    print(args.start_from)
    model = torch.load(args.start_from)
    config = model.config
    model = model.to(device=device)
    test_loader = DataLoader(test_data, 1, shuffle=False)


    it_num = 0
    result = []
    lfile = open(args.rundir + '/test_result.txt', 'w')
    for test_input in test_loader:
        it_num += 1
        test_input = test_input.to(device)
        test_output = model(test_input)
        test_output.detach()
        lfile.write('the {}th gene, test output = {}\n'.format(it_num, test_output))
        #print("order=",it_num,test_output)
        result.append(test_output)
    print(it_num)





