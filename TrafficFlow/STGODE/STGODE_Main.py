import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from lib.evaluate import MAE_torch
from STGODE_Config import args
from STGODE_Trainer import Trainer
from STGODE_Utils import *
from model.STGODE.stgode import ODEGCN as Network


def load_data(args):
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, 
                                                                            normalizer=args.normalizer, 
                                                                            tod=False,
                                                                            dow=False, 
                                                                            single=False)
    dtw_matrix, sp_matrix = read_data(args)
    A_sp_wave = get_normalized_adj(sp_matrix).to(args.device)
    A_se_wave = get_normalized_adj(dtw_matrix).to(args.device)
    return A_sp_wave, A_se_wave, train_dataloader, val_dataloader, test_dataloader, scaler


def generate_model_components(args, A_sp_hat, A_se_hat):
    # 1. model
    model = Network(
        num_nodes=args.num_node,
        num_features=args.input_dim,
        num_timesteps_input=args.window,
        num_timesteps_output=args.horizon,
        A_sp_hat=A_sp_hat,
        A_se_hat=A_se_hat
    )
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    # print the number of model parameters
    print_model_parameters(model, only_num=False)
    # 2. loss
    def masked_mae_loss(scaler, mask_value):
        def loss(preds, labels):
            if scaler:
                preds = scaler.inverse_transform(preds)
                labels = scaler.inverse_transform(labels)
            mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
            return mae
        return loss
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss = torch.nn.SmoothL1Loss().to(args.device)
    elif args.loss_func == 'huber':
        loss = torch.nn.HuberLoss(delta=1.0).to(args.device)
    else:
        raise ValueError
    # 3. optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr_init)
    # 4. learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        # lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
        #                                                     milestones=lr_decay_steps,
        #                                                     gamma=args.lr_decay_rate)
    return model, loss, optimizer, lr_scheduler


def get_log_dir(model, dataset, debug):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))  # /GNN4Traffic/TrafficFlow
    log_dir = os.path.join(current_dir,'log', model, dataset, current_time) 
    if os.path.isdir(log_dir) == False and not debug:
        os.makedirs(log_dir, exist_ok=True)  # run.log
    return log_dir


if __name__ == '__main__':
    init_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'
    A_sp_wave, A_se_wave, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset, args.debug)
    model, loss, optimizer, lr_scheduler = generate_model_components(args, A_sp_wave, A_se_wave)
    trainer = Trainer(
        args=args, 
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
        checkpoint = "../log/STGODE/PEMSD4/20230211215341/PEMSD4_STGODE_best_model.pth"
        trainer.test(model, args, test_dataloader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError