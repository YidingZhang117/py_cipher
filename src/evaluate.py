import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataloader import Cipher_Dataloader
from network import Network
from config import DATA_FOLDER #import a variable


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='simple_network', help='model file definition')
    parser.add_argument('-bs',default=8, type=int, help='batch size')
    parser.add_argument('-lt', default=10, type=int, help='Loss file saving refresh interval (seconds)')
    parser.add_argument('-lr', default=1e-3 , type= float, help='Learning rate')
    parser.add_argument('-data_path', default='data/', help='Training path')
    parser.add_argument('-rundir', default='../results/test' , help='Running directory')
    parser.add_argument('-ep', default=10 , type=int , help='Epochs')
    parser.add_argument('-start_from', default='' , help='Start from previous model')
    parser.add_argument('-optim', default='SGD', help='choose the optimizer')
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

    train_data = Cipher_Dataloader(full_path, "train")
    test_data = Cipher_Dataloader(full_path, "test")
    return train_data, test_data

def evaluate(loader, model, crit, device):
    model.eval()
    total_loss = 0.0
    i = 0
    count = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for val_pos_input, val_pos_target, val_neg_input, val_neg_target in loader:
        val_input = torch.cat([val_pos_input,val_neg_input])
        val_target = torch.cat([val_pos_target,val_neg_target])
        val_input, val_target = val_input.to(device), val_target.to(device)
        val_output = model(val_input)
        val_output.detach()
        loss = crit.forward(val_output, val_target)
        total_loss += loss.item()
        i += 1
        # calculate accuracy, recall and precision
        if torch.argmax(val_output[]) == torch.argmax(val_target):
            count += 1
            if torch.argmax(val_output) == 0: #ture positive
               true_positive += 1
        else:
            print("Wrong pair:",val_output,"\n", val_target)
            if torch.argmax(val_output) == 0: #false positive
                false_positive += 1
            if torch.argmax(val_output) == 1: #false negative
                false_negative += 1
            # print(torch.argmax(val_output).item(), torch.argmax(val_target).item())
            # print(val_input)

    print("test accuracy = ", count/i)
    print("test recall = ",true_positive/(true_positive+false_negative))
    print("test recall = ", true_positive/(true_positive+false_positive))
    return total_loss/i

if __name__ == '__main__':
    # --- arguments ---
    args = parseArgs()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # --- dataloader ---
    try:
        train_data, test_data = get_dataloader(args)
    except Exception as e:
        print("Error when get dataloader")
        sys.exit(1)

    # run path
    if not os.path.exists(args.rundir):
        os.mkdir(args.rundir)
    torch.save(args, args.rundir + '/args.pth')

    # --- Model and criterion ---
    crit = nn.MSELoss()
    # model
    config = {}
    if args.start_from != '':  # from previous checkpoint
        print(args.start_from)
        model = torch.load(args.start_from)
        if model.period is None:
            model.period = 1
        model.period += 1
        config = model.config
    else:
        model = Network()
        # print([i for i in model.modules()])
        model.period = 1
    config['learningRate'] = args.lr
    model = model.to(device=device)
    # optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.6)  # optimizer
        print('Using SGD')
    test_loader = DataLoader(test_data, 1, shuffle=False)
    valid_eval_loss = evaluate(test_loader, model, crit, device)