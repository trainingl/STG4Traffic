import sys
sys.path.append('../')

import os
import torch
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from TGCN_Config import args
from TGCN_Trainer import Trainer
from model.TGCN.tgcn import TGCN as Network


def load_data(args):
    data_loader = load_dataset(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data_loader['scaler']
    # 加载拓扑图的邻接矩阵
    _, _, adj_mx = load_pickle(args.graph_pkl)
    adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    return adj_mx, data_loader, scaler

def mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((inputs - targets) ** 2) / 2
    return mse_loss + reg_loss

def generate_model_components(args, adj_mx):
    adj_mx = torch.tensor(adj_mx).to(args.device)
    # 1. model
    model = Network(
        adj_mx=adj_mx, 
        input_dim=args.input_dim, 
        hidden_dim=args.hidden_dim, 
        out_dim=args.horizon
    )
    model = model.to(args.device)
    # print the number of model parameters
    print_model_parameters(model, only_num=False)
    # 2. loss
    if args.loss_func == 'mask_mae':
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
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
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
    supports, data_loader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset)
    model, loss, optimizer, lr_scheduler = generate_model_components(args, supports)
    trainer = Trainer(
        args=args, 
        data_loader=data_loader, 
        scaler=scaler, 
        model=model, 
        loss=loss, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        cl=args.cl, 
        new_training_method=args.new_training_method
    )
    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/TGCN/METRLA/20230104184903/METRLA_TGCN_best_model.pth"
        trainer.test(args, model, data_loader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError