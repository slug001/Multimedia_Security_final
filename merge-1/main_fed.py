#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from models.test import test_img
from models.Fed import FedAvg
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model, vgg11
from models.resnet20 import resnet20
from models.Update import LocalUpdate
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.defense import  get_update, flame, crowdguard
from utils.text_helper import TextHelper
from models.Attacker import attacker
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import math
import yaml
import datetime
from utils.text_load import *


matplotlib.use('Agg')

"""
作用： 將實驗過程中的主要任務準確率 (accu_list) 和後門任務成功率 (back_list) 以及實驗設定 (args) 寫入到指定的檔案 (filename) 中。
analyse=True 時： 計算並寫入最後 10% 回合的最佳準確率 (max acc)、平均後門成功率 (ABSR) 和最佳後門成功率 (BBSR)。
"""
def write_file(filename, accu_list, back_list, args, analyse=False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("backdoor_accuracy=")
    f.write(str(back_list))
    if args.defence == "krum":
        krum_file = filename + "_krum_dis"
        torch.save(args.krum_distance, krum_file)
    if args.defence == "flare":
        benign_file = filename + "_benign_dis.torch"
        malicious_file = filename + "_malicious_dis.torch"
        torch.save(args.flare_benign_list, benign_file)
        torch.save(args.flare_malicious_list, malicious_file)
        f.write('\n')
        f.write("avg_benign_list=")
        f.write(str(np.mean(args.flare_benign_list)))
        f.write('\n')
        f.write("avg_malicious_list=")
        f.write(str(np.mean(args.flare_malicious_list)))
    if analyse == True:
        need_length = len(accu_list) // 10
        acc = accu_list[-need_length:]
        back = back_list[-need_length:]
        best_acc = round(max(acc), 2)
        average_back = round(np.mean(back), 2)
        best_back = round(max(back), 2)
        f.write('\n')
        f.write('BBSR:')
        f.write(str(best_back))
        f.write('\n')
        f.write('ABSR:')
        f.write(str(average_back))
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc, average_back, best_back
    f.close()


"""
作用： 從給定的資料集 (dataset) 中隨機（IID）抽取指定數量 (dataset_size) 的樣本，用於構建一個中央伺服器可能擁有的（用於某些防禦如 FLTrust 的）小部分驗證資料集。
注意： 目前只保留 FLAME 的設定下，這個函式可能不會被直接調用，因為 FLAME 的原始版本不依賴伺服器端的特定根資料集。
"""
def central_dataset_iid(dataset, dataset_size):
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset


"""
作用： 檢查指定路徑 (path) 的資料夾是否存在，如果不存在則創建它。用於確保儲存結果的資料夾存在。
"""
def test_mkdir(path):
    if not os.path.isdir(path):
        #os.mkdir(path)
        os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    # 將 lp_attack（可能是 Layer Poisoning Attack 的縮寫）映射到 adaptive 攻擊模式，表明您的 BC 層攻擊的核心是自適應的。
    if args.attack == 'lp_attack':
        args.attack = 'adaptive'  # adaptively control the number of attacking layers
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    test_mkdir('./' + args.save)
    print_exp_details(args)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion_mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            # dict_users = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
            dict_users = cifar_noniid([x[1] for x in dataset_train], args.num_users, 10, args.p)
            print('main_fed.py line 137 len(dict_users):', len(dict_users))
    elif args.dataset == 'reddit':
        with open(f'./utils/words.yaml', 'r') as f:
            params_loaded = yaml.safe_load(f)
        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        helper = TextHelper(
            current_time=current_time,
            params=params_loaded,
            name=params_loaded.get('name', 'text'),
        )
        helper.load_data()
        dataset_train = helper.train_data
        dict_users = cifar_iid(dataset_train, args.num_users)
        dataset_test = helper.test_data
        args.helper = helper
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'VGG' and args.dataset == 'cifar':
        net_glob = vgg19_bn().to(args.device)
    elif args.model == 'VGG11' and args.dataset == 'cifar':
        net_glob = vgg11().to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == "resnet20" and args.dataset == 'cifar':
        net_glob = resnet20().to(args.device)
    elif args.model == "rlr_mnist" or args.model == "cnn":
        net_glob = get_model('fmnist').to(args.device)
    elif args.model == 'lstm':
        helper.create_model()
        net_glob = helper.local_model.to(args.device)
    else:
        exit('Error: unrecognized model')
   
    
    # 將模型設置為訓練模式。
    net_glob.train()

    # copy weights 獲取全域模型的初始權重。
    w_glob = net_glob.state_dict()

    # training
    loss_train = []

    # 為 FLAME 防禦準備的列表，用於儲存距離/分數。
    args.flare_benign_list=[]
    args.flare_malicious_list=[]

    # 設定一個準確率閾值 (0)，只有當模型的良性任務準確率超過這個閾值時，才開始執行後門攻擊。
    if math.isclose(args.malicious, 0):
        backdoor_begin_acc = 100
    else:
        backdoor_begin_acc = args.attack_begin  # overtake backdoor_begin_acc then attack
    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)  # get root dataset for FLTrust
    base_info = get_base_info(args)
    filename = './' + args.save + '/accuracy_file_{}.txt'.format(base_info)  # 生成用於記錄結果的檔案名。 log hyperparameters

    if args.init != 'None':  # continue from checkpoint
        param = torch.load(args.init)
        net_glob.load_state_dict(param)
        print("load init model")

    val_acc_list = [0.0001]  # Acc list
    backdoor_acculist = [0]  # BSR list

    args.attack_layers = []  # keep LSA  關鍵變數，用於儲存（並在多輪之間傳遞）LSA 識別出的後門關鍵層 (BC layers)。


    if args.log_distance == True:
        args.krum_distance = []
        args.krum_layer_distance = []
    malicious_list = []  # list of the index of malicious clients
    for i in range(int(args.num_users * args.malicious)):
        malicious_list.append(i)

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)] # 存儲客戶端訓練後的完整模型權重字典列表；

    for iter in range(args.epochs):
        loss_locals = []   # 存儲本輪所有參與客戶端的本地訓練損失。
        if not args.all_clients:
            w_locals = []
            w_updates = [] # 存儲客戶端模型權重相對於上一輪全域模型的更新量列表。
        
        m = max(int(args.frac * args.num_users), 1)  # number of clients in each round
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select the clients for a single round

        if backdoor_begin_acc < val_acc_list[-1]:  # start attack only when Acc overtakes backdoor_begin_acc
            backdoor_begin_acc = 0
            attack_number = int(args.malicious * m)  # number of malicious clients in a single round
        else:
            attack_number = 0
            
        if args.scaling_attack_round != 1:
            # scaling attack begin 100-th round and perform each args.attack_round round
            if iter > 100 and iter%args.scaling_attack_round == 0:
                attack_number = attack_number
            else:
                attack_number = 0
        mal_weight=[]
        mal_loss=[]
        for num_turn, idx in enumerate(idxs_users):
            if attack_number > 0:  # upload models for malicious clients
                args.iter = iter
                m_idx = None
                
                """
                執行後門攻擊（包括 LSA 和自適應 BC 層攻擊）的核心步驟。
                它會返回惡意客戶端生成的模型權重列表 (mal_weight)、損失和更新後的 args.attack_layers。
                """
                mal_weight, loss, args.attack_layers = attacker(malicious_list, attack_number, args.attack, dataset_train, dataset_test, dict_users, net_glob, args, idx = m_idx)
                attack_number -= 1
                w = mal_weight[0]  # 取第一個惡意模型權重
            else:  # upload models for benign clients
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(
                    net=copy.deepcopy(net_glob).to(args.device))
            w_updates.append(get_update(w, w_glob)) # 計算並儲存模型更新。
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w)) # 儲存完整的本地模型權重。
            loss_locals.append(copy.deepcopy(loss)) # 儲存本地損失。

        if args.defence == 'avg':  # no defence
            w_glob = FedAvg(w_locals)
        elif args.defence == 'flame':
            w_glob = flame(w_locals, w_updates, w_glob, args, debug=args.debug)
        elif args.defence == 'crowdguard':
            crowdguard(w_locals, w_updates, w_glob, args, debug=args.debug)
        else:
            print("Wrong Defense Method")
            os._exit(0)

        # copy weight to net_glob 將聚合後的新權重載入到全域模型中。
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if iter % 1 == 0:
            acc_test, _, back_acc = test_img(
                net_glob, dataset_test, args, test_backdoor=True)
            print("Main accuracy: {:.2f}".format(acc_test))
            print("Backdoor accuracy: {:.2f}".format(back_acc))
            if args.model == 'lstm':
                val_acc_list.append(acc_test)
            else:
                val_acc_list.append(acc_test.item())

            backdoor_acculist.append(back_acc)
            write_file(filename, val_acc_list, backdoor_acculist, args)

    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)

    # plot loss curve
    plt.figure()
    plt.xlabel('communication')
    plt.ylabel('accu_rate')
    plt.plot(val_acc_list, label='main task(acc:' + str(best_acc) + '%)')
    plt.plot(backdoor_acculist, label='backdoor task(BBSR:' + str(bbsr) + '%, ABSR:' + str(absr) + '%)')
    plt.legend()
    title = base_info
    plt.title(title)
    plt.savefig('./' + args.save + '/' + title + '.pdf', format='pdf', bbox_inches='tight')

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    torch.save(net_glob.state_dict(), './' + args.save + '/model' + '.pth')
