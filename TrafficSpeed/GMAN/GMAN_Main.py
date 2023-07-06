import sys
sys.path.append('../')

import os
import torch
from datetime import datetime
from lib.utils import *
from GMAN_Utils import *
from GMAN_Config import args
from GMAN_Trainer import Trainer
from model.GMAN.gman import GMAN as Network


def load_data(args):
    data, scaler = loadData(args)
    return data, scaler

def generate_model_components(args):
    # 1. model
    model = Network(
        L = args.L,
        K = args.K,
        d = args.d,
        num_his = args.window,
        bn_decay = args.bn_decay,
        steps_per_day = args.steps_per_day,
        use_bias = args.use_bias,
        mask = args.mask
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
    data_loader, scaler = load_data(args)
    args.SE = torch.tensor(data_loader['SE']).to(args.device)
    args.log_dir = get_log_dir(args.model, args.dataset)
    model, loss, optimizer, lr_scheduler = generate_model_components(args)
    trainer = Trainer(
        args=args, 
        data_loader=data_loader, 
        scaler=scaler, 
        model=model, 
        loss=loss, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler
    )
    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/GMAN/METRLA/20221210001716/METRLA_GMAN_best_model.pth"
        trainer.test(args, model, data_loader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError