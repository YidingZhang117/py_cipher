import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy

from dataloader import Cipher_Dataloader
from network import Network
from config import DATA_FOLDER #import a variable


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='simple_network', help='model file definition')
    parser.add_argument('-bs',default=4, type=int, help='batch size')
    parser.add_argument('-lt', default=10, type=int, help='Loss file saving refresh interval (seconds)')
    parser.add_argument('-lr', default=4e-4, type= float, help='Learning rate')
    parser.add_argument('-data_path', default='data/', help='Training path')
    parser.add_argument('-rundir', default='../results/test' , help='Running directory')
    parser.add_argument('-ep', default=50, type=int , help='Epochs')
    # parser.add_argument('-start_from', default='../results/1216_best1/Best_model_period1.pth' , help='Start from previous model')
    parser.add_argument('-start_from', default='', help='Start from previous model')
    parser.add_argument('-optim', default='SGD', help='choose the optimizer')
    args = parser.parse_args()
    return args


def save_model(model, directory, current_ep, config):
    model.config = config
    torch.save(model, directory+'/model_period'+str(model.period)+'_'+str(current_ep)+'.pth')


def save_best_model(model, directory, config, current_ep):
    model.config = config
    model.iter = current_ep
    torch.save(model, os.path.join(directory,'Best_model_period'+str(model.period)+'.pth'))


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

    pos_ind = [ind for ind in range(15)]
    neg_ind = [ind for ind in range(80)]
    train_ind = [pos_ind, neg_ind]
    test_ind = [[15,16], [80,81,82,83]]

    train_data = Cipher_Dataloader(full_path, "train", train_ind)
    test_data = Cipher_Dataloader(full_path, "test", test_ind)
    return train_data, test_data


def evaluate(loader, model, crit, device):
    #print the weight
    weight = [content for content in model.modules()]
    #print(dir(weight[2])) # look at the attribute of this class
    #print("linear1:", weight[2].weight)
    # print("linear2:", weight[4].weight)
    # print("linear3:", weight[6].weight)
    # print("linear4:", weight[8].weight)
    # print("linear5:", weight[10].weight)
    model.eval()

    total_loss = 0.0
    i = 0
    count = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for val_input, val_target in loader:
        # print("input:",val_input,val_target)
        val_input, val_target = val_input.to(device), val_target.to(device)
        val_output = model(val_input)
        val_output.detach()
        # val_target_index = []
        # for ii in val_target:
        #     #print("val_target: ",ii)
        #     class_i = torch.argmax(ii)
        #     i_val = class_i.numpy()
        #     i_list = i_val.tolist()
        #     val_target_index.append(i_list)
        # val_target_index = torch.tensor(val_target_index)
        #print("val_target_index: ",val_target_index )
        #print("val_output: ",val_output )
        loss = crit.forward(val_output, val_target)
        # print(val_output)
        # print( val_output)
        # print("loss:",loss)

        total_loss += loss.item()
        # print("loss:",loss.item())
        i += 1
        # calculate accuracy, recall and precision
        if torch.argmax(val_output) == torch.argmax(val_target):
            count += 1
            if torch.argmax(val_output) == 0: #ture positive
                true_positive += 1
            else:
                true_negative += 1
            print("Right pair:", val_output, val_target)
        else:
            print("Wrong pair:",val_output, val_target)
            if torch.argmax(val_output) == 0: #false positive
                false_positive += 1
            if torch.argmax(val_output) == 1: #false negative
                false_negative += 1
            # print(torch.argmax(val_output).item(), torch.argmax(val_target).item())
            # print(val_input)
    print(i)
    print("true positive:",true_positive)
    print("true negative:",true_negative)
    print("false negative:",false_negative)
    print("false positive:",false_positive)
    recall = true_positive/(true_positive+false_negative)
    if (true_positive+false_positive)!=0:
        precision = true_positive/(true_positive+false_positive)
    else:
        precision = 'NA'
    print("test accuracy = ", count/i)
    print("test recall = ",recall)
    print("test precision = ", precision)
    print("total_loss",total_loss)
    # print("F = ", 2*recall*precision/(recall+precision))
    return total_loss/i


def train(args, config, train_data, test_data, model, crit, optimizer, scheduler, device):
    lfile = open(args.rundir+'/training_loss_period'+str(model.period)+'.txt', 'w')
    best_valist_set_loss = 100.0
    total_loss = 0.0
    ratio = 2
    for i in range(args.ep):
        scheduler.step()
        model.train(True)
        # get the train dataset
        train_loader = DataLoader(train_data, args.bs, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, 1, shuffle=False)
        # per batch train
        it_num = 0
        for batch_pos_input, batch_pos_target, batch_neg_input, batch_neg_target in train_loader:
            optimizer.zero_grad()
            #get pos output and pos loss
            batch_pos_input, batch_pos_target = batch_pos_input.to(device), batch_pos_target.to(device)
            batch_pos_output = model(batch_pos_input)
            # batch_pos_target_index = []
            # for m in batch_pos_target:
            #     # print(i)
            #     class_i = torch.argmax(m)
            #     i_val = class_i.numpy()
            #     i_list = i_val.tolist()
            #     batch_pos_target_index.append(i_list)
            # batch_pos_target_index = torch.tensor(batch_pos_target_index)
            # batch_pos_loss = crit(batch_pos_output, batch_pos_target_index)
            batch_pos_loss = crit(batch_pos_output, batch_pos_target)
            #batch_pos_loss = crit(batch_pos_output, batch_pos_target)#for MSELoss
            #get neg output and neg loss
            batch_neg_input, batch_neg_target = batch_neg_input.to(device), batch_neg_target.to(device)
            batch_neg_output = model(batch_neg_input)
            # batch_neg_target_index = []
            # for ii in batch_neg_target:
            #     # print(i)
            #     class_i = torch.argmax(ii)
            #     i_val = class_i.numpy()
            #     i_list = i_val.tolist()
            #     batch_neg_target_index.append(i_list)
            # batch_neg_target_index = torch.tensor(batch_neg_target_index)
            # batch_neg_loss = crit(batch_neg_output, batch_neg_target_index)
            batch_neg_loss = crit(batch_neg_output, batch_neg_target)
            # print(batch_pos_loss.item())
            # print(batch_neg_loss.item())
            batch_loss = ratio * batch_pos_loss + batch_neg_loss
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            # if it_num % 20 == 0:
            #     t_now = datetime.datetime.now()
            #     t = t_now.strftime("%Y-%m-%d %H:%M:%S")
            #     print(t)
            # print(('iteration {}, loss = {}'.format(it_num, batch_loss.item())))
            lfile.write('iteration {}, loss = {}\n'.format(it_num, batch_loss.item()))
            it_num += 1

        print('================================================================')
        print('Evaluating at epoch {}'.format(i))
        valid_eval_loss = evaluate(test_loader, model, crit, device)
        print("valid_eval_loss:", valid_eval_loss)
        # save best model
        if best_valist_set_loss > valid_eval_loss:
            best_valist_set_loss = valid_eval_loss
            save_best_model(model, args.rundir, config, i)


# main 
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
    torch.save(args, args.rundir+'/args.pth')

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

    # weight = [content for content in model.modules()]
    # print(dir(weight[2])) # look at the attribute of this class
    #print("linear1:", weight[2].weight)
    # print("linear2:", weight[4].weight)
    # print("linear3:", weight[6].weight)
    # print("linear4:", weight[8].weight)
    # print("linear5:", weight[10].weight)

    # optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #optimizer
        print('Using SGD')
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
    train(args, config, train_data, test_data, model, crit, optimizer, scheduler, device)
    