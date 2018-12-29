import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np

from dataloader import Cipher_Dataloader
from network import Network
from config import DATA_FOLDER #import a variable


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='simple_network', help='model file definition')
    parser.add_argument('-bs',default=4, type=int, help='batch size')
    parser.add_argument('-lt', default=10, type=int, help='Loss file saving refresh interval (seconds)')
    parser.add_argument('-lr', default=5e-5, type= float, help='Learning rate')
    parser.add_argument('-data_path', default='data/', help='Training path')
    parser.add_argument('-rundir', default='../results/test' , help='Running directory')
    parser.add_argument('-ep', default=10, type=int , help='Epochs')
    #parser.add_argument('-start_from', default='../results/1227best3_newloss0.0012075659663726885/Best_model_period2.pth' , help='Start from previous model')
    parser.add_argument('-start_from', default='', help='Start from previous model')
    #../results/1125best3_0.019168664837100852/Best_model_period1.pth
    parser.add_argument('-optim', default='SGD', help='choose the optimizer')
    #parser.add_argument('-choose_ind',default='../results/1227best2_newloss0.0014167820695642301/choose_ind.pth',help='define index of train and test sample')
    parser.add_argument('-choose_ind',default='',help='define index of train and test sample')
    args = parser.parse_args()
    return args


def save_model(model, directory, current_ep, config):
    model.config = config
    torch.save(model, directory+'/model_period'+str(model.period)+'_'+str(current_ep)+'.pth')


def save_best_model(model, directory, config, current_ep):
    model.config = config
    model.iter = current_ep
    print('===============================')
    print(os.path.join(directory,'Best_model_period'+str(model.period)+'.pth'))
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
    if args.choose_ind == '':
        all_pos_ind = [i for i in range(0,17)]
        all_neg_ind = [i for i in range(0,84)]
        pos_ind = random.sample(range(0, 17),15 )
        neg_ind = random.sample(range(0,84),80)
        #pos_ind = [ind for ind in range(15)]
        #neg_ind = [ind for ind in range(80)]
        train_ind = [pos_ind, neg_ind]
        test_pos_ind = list(set(all_pos_ind).difference(set(pos_ind)))
        test_neg_ind = list(set(all_neg_ind).difference(set(neg_ind)))
        test_ind = [test_pos_ind,test_neg_ind]
        print("test ind:")
        print(test_ind)
        #test_ind = [[15,16], [80,81,82,83]]
    else:
        print(args.choose_ind)
        # print(torch.load(args.choose_ind))
        train_ind, test_ind = torch.load(args.choose_ind)
        print(test_ind)
    train_data = Cipher_Dataloader(full_path, "train", train_ind,test_ind)
    test_data = Cipher_Dataloader(full_path, "test", train_ind, test_ind)
    return train_data, test_data,train_ind,test_ind


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
    eval_ratio = 2
    eval_result_epoch = []
    for val_input, val_target in loader:
        # print("input:",val_target)
        # print(torch.argmax(val_target))
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
        if torch.argmax(val_target) == torch.tensor(0):
            #print(val_target)
            #print("positive")
            loss = eval_ratio * crit.forward(val_output, val_target)
        else:
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
            eval_result = ["Right pair:",val_output, val_target]
            eval_result_epoch.append(eval_result)
            #print(eval_result)
           # print("Right pair:", val_output, val_target)
        else:
            eval_result = ["Wrong pair:",val_output, val_target]
            eval_result_epoch.append(eval_result)
            #print(eval_result)
            #print("Wrong pair:",val_output, val_target)
            if torch.argmax(val_output) == 0: #false positive
                false_positive += 1
            if torch.argmax(val_output) == 1: #false negative
                false_negative += 1
            # print(torch.argmax(val_output).item(), torch.argmax(val_target).item())
            # print(val_input)
    for row in eval_result_epoch:
        print(row)
    #print(eval_result_epoch)
    # print(i)
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
    return total_loss/i,eval_result_epoch


def train(args, config, train_data, test_data, model, crit, optimizer, scheduler, device):
    lfile = open(args.rundir+'/training_loss_period'+str(model.period)+'.txt', 'w')
    best_valist_set_loss = 100.0
    total_loss = 0.0
    ratio = 3
    for i in range(args.ep):
        scheduler.step()
        model.train(True)
        # get the train dataset
        train_loader = DataLoader(train_data, args.bs, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, 1, shuffle=False)
        # per batch train
        it_num = 0
        for batch_pos_input, batch_pos_target, batch_neg_input, batch_neg_target in train_loader:
            it_num += 1
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
            # print("training result:=====================")
            # print("epoch:",i)
            # print("iteration:",it_num)
            # for ind in range(len(batch_neg_output)):
            #     # if torch.argmax(batch_neg_output[ind]) == torch.argmax(batch_neg_target[ind]):
            #     #     print("Right:",batch_neg_output[ind],batch_neg_target[ind])
            #     # else:
            #     #     print("Wrong:",batch_neg_output[ind],batch_neg_target[ind])
            #     if torch.argmax(batch_pos_output[ind]) == torch.argmax(batch_pos_target[ind]):
            #         print("Right:", batch_pos_output[ind], batch_pos_target[ind])
            #     else:
            #         print("Wrong:", batch_pos_output[ind], batch_pos_target[ind])
        print('================================================================')
        print('Evaluating at epoch {}'.format(i))
        valid_eval_loss,eval_result_epoch = evaluate(test_loader, model, crit, device)
        print("valid_eval_loss:", valid_eval_loss)
        # save best model
        if best_valist_set_loss > valid_eval_loss:
            best_valist_set_loss = valid_eval_loss
            best_eval_result_epoch = eval_result_epoch
            best_epoch = i
            save_best_model(model, args.rundir, config, i)
    return best_valist_set_loss, best_eval_result_epoch, best_epoch


# main 
if __name__ == '__main__':
    # --- arguments ---
    args = parseArgs()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    best_loss = []
    best_epoch_all = []
    eval_result_all = []
    test_ind_all = []
    for sample_i in range(0,1):
        # --- dataloader ---
        try:
            train_data, test_data,train_ind, test_ind = get_dataloader(args)
        except Exception as e:
            print("Error when get dataloader")
            sys.exit(1)

        # run path
        if not os.path.exists(args.rundir):
            os.mkdir(args.rundir)
        torch.save(args, args.rundir+'/args.pth')
        torch.save((train_ind, test_ind), os.path.join(args.rundir, 'choose_ind.pth'))

        # --- Model and criterion ---
        crit = nn.MSELoss()
        # model
        config = {}
        if args.start_from != '':  # from previous checkpoint
            print(args.start_from)
            model = torch.load(args.start_from)
            #print(type(model))
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

        #weight = [content for content in model.modules()]
        #print(dir(weight[2])) # look at the attribute of this class
        #print("linear1:", weight[2].weight)
        # print("linear2:", weight[4].weight)
        # print("linear3:", weight[6].weight)
        # print("linear4:", weight[8].weight)
        # print("linear5:", weight[10].weight)

        # optimizer
        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #optimizer
            print('Using SGD')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        best_valist_set_loss,eval_result_epoch,best_epoch = train(args, config, train_data, test_data, model, crit, optimizer, scheduler, device)
        best_loss.append(best_valist_set_loss)
        best_epoch_all.append(best_epoch)
        eval_result_all.append(eval_result_epoch)
        test_ind_all.append(test_ind)

    best_loss_np = np.array(best_loss,dtype=float)
    mean_loss = np.mean(best_loss_np)
    sample_i = 0
    for sample in eval_result_all:
        print("=======sample time %d:========" % (sample_i))
        print("best epoch at",best_epoch_all[sample_i])
        print("test ind:",test_ind_all[sample_i])
        for row in sample:
            print(row)
        sample_i += 1

    print("best loss in all sampling time:")
    print(best_loss_np)
    print("mean loss:")
    print(mean_loss)


