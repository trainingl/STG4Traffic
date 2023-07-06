import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from DMSTGCN_Config import args
from DMSTGCN_Utils import *
from DMSTGCN_Trainer import Trainer
from model.DMSTGCN.dmstgcn import DMSTGCN as Network


def load_data(args):
    data = load_dataset(
        args, 
        batch_size=args.batch_size, 
        valid_batch_size=args.batch_size, 
        test_batch_size=args.batch_size, 
        days=288
    )
    train_dataloader = data['train_loader']
    val_dataloader = data['val_loader']
    test_dataloader = data['test_loader']
    scaler = data['scaler']
    return data, train_dataloader, val_dataloader, test_dataloader, scaler


def generate_model_components(args):
    # 1. model
    model = Network(
        device=args.device,
        num_nodes=args.num_node, 
        dropout=args.dropout, 
        out_dim=args.horizon, 
        residual_channels=args.hidden_dim, 
        dilation_channels=args.hidden_dim, 
        end_channels=args.end_channels, 
        kernel_size=args.kernel_size, 
        blocks=args.blocks, 
        layers=args.layers, 
        days=args.days, 
        dims=args.node_dim, 
        order=args.order, 
        in_dim=args.in_dim, 
        normalization=args.normalization
    )
    model = model.to(args.device)
    # print the number of model parameters
    print_model_parameters(model, only_num=False)
    # 2. loss
    if args.loss_func == 'masked_mae':
        loss = masked_mae
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss = torch.nn.SmoothL1Loss().to(args.device)
    else:
        raise ValueError
    # 3. optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    # 4. learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
    return model, loss, optimizer, lr_scheduler


def get_log_dir(model, dataset):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))  # /GNN4Traffic/TrafficSpeed
    log_dir = os.path.join(current_dir,'log', model, dataset, current_time) 
    return log_dir


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'
    data, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset)
    model, loss, optimizer, lr_scheduler = generate_model_components(args)
    
    trainer = Trainer(
        args=args, 
        data=data,
        train_loader=train_dataloader, 
        val_loader=val_dataloader, 
        test_loader=test_dataloader, 
        scaler=scaler, 
        model=model, 
        loss=loss, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler
    )
    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/DMSTGCN/PEMSD4/20221224183122/PEMSD4_DMSTGCN_best_model.pth"
        trainer.test(args, model, data, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError