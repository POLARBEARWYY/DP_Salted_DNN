#_# Python Libraries 
import os
import argparse
import numpy as np
import torch
import torchvision
from torchinfo import summary
import torch.optim as optim
import torch.nn as nn

#_# Our Source Code
from src import datasets, utils
from src.models import simple_cnn, wide_resnet
import exp_setup

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# 添加RDP机制需要添加以下内容
from opacus import PrivacyEngine

# 定义并添加RDP机制
def add_rdp_mechanism(model, optimizer, epsilon, delta, max_grad_norm):
    privacy_engine = PrivacyEngine(
        module=model,
        sample_rate=0.01,  # 这是一个示例，你需要根据你的数据集和批量大小进行调整
        noise_multiplier=np.sqrt(2 * np.log(1.25 / delta)) / epsilon,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)
    return privacy_engine

if __name__ == "__main__":

    args = exp_setup.get_parser().parse_args()

    args.device = "cpu"
    if torch.cuda.is_available():
        args.device = "cuda"        
        torch.cuda.set_device(args.gpu_id)
    print(torch.cuda.get_device_properties(args.device))
    
    if args.reproducibility:
        torch.manual_seed(args.rand_seed)
        torch.cuda.manual_seed(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.rand_seed)

    if args.split == 0:
        train_data, train_labels,  test_data,  test_labels = datasets.get_dataset(args, verbose=1)
        
        # 应用差分隐私预处理
        #epsilon = 1000  # 设置差分隐私参数
        #train_data, test_data = exp_setup.preprocess_data_with_dp(train_data, test_data, epsilon)
        
        dataset = ((train_data, train_labels),(None,None), (test_data,  test_labels))      
    else:
        raise Exception("Setting of split == 1 is not implemented in this version")    
        # train_data, train_labels, valid_data, valid_labels, test_data,  test_labels = datasets.get_dataset(args, verbose=1)
        # dataset = ((train_data, train_labels), (valid_data, valid_labels), (test_data,  test_labels))  
    
        
    # For LeNet
    model = simple_cnn.SimpleCNN(num_classes=args.num_classes, salt_layer=args.salt_layer,
                                mean =  datasets.CIFAR10_MEAN, 
                                std = datasets.CIFAR10_STD, 
                                num_input_channels=args.num_input_channels)

    ## For WideResNet
    # model = wide_resnet.WideResNet(num_classes = args.num_classes,
    #                                     width = 3, 
    #                                     mean =  datasets.CIFAR10_MEAN, 
    #                                     std = datasets.CIFAR10_STD, 
    #                                     num_input_channels=args.num_input_channels)
    ## For SaltedWideResNet
    # model = wide_resnet.SaltyWideResNet(num_classes = args.num_classes,
    #                                     width = 3, 
    #                                     mean =  datasets.CIFAR10_MEAN, 
    #                                     std = datasets.CIFAR10_STD, 
    #                                     num_input_channels=args.num_input_channels)

    ## For ConvNet (PAMAP2)
    # model = simple_cnn.SenNet(salt_layer=args.salt_layer)
        
    model.to(args.device)
    if args.dataset == "cifar10":
        if args.salt_layer == -1:
            summary(model, [(1, args.num_input_channels, 32, 32)], device=args.device)       
        elif 0<= args.salt_layer <=5: 
            summary(model, [(1, args.num_input_channels, 32, 32),(1,1,1,1)], device=args.device)       
        else:
            summary(model, [(1, args.num_input_channels, 32, 32),(1,args.num_classes)], device=args.device)       
    elif args.dataset == "pamap":
        summary(model, [(1, 1, 27, 200),(1,1,1,1)], device=args.device)  

    
    # Define your optimizer here
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 添加RDP机制
    epsilon = 1.0  # 根据需求设置
    delta = 1e-5   # 根据需求设置
    max_grad_norm = 1.0  # 根据需求设置
    privacy_engine = add_rdp_mechanism(model, optimizer, epsilon, delta, max_grad_norm)

    utils.train_test(args, model, dataset, save_model=True)

    # 输出隐私消耗
    epsilon_spent, best_alpha = privacy_engine.get_privacy_spent(delta)
    print(f"Privacy spent: (ε = {epsilon_spent:.2f}, δ = {delta}) at alpha = {best_alpha}")
