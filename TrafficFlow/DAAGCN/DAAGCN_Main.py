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
from DAAGCN_Config import args
from DAAGCN_Trainer import Trainer
from model.DAAGCN.generator import DAAGCN as Generator
from model.DAAGCN.discriminator import Discriminator, Discriminator_RF


def load_data(args):
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, 
                                                                            normalizer=args.normalizer, 
                                                                            tod=False,
                                                                            dow=False, 
                                                                            single=False)
    if args.graph_type == 'BINARY':
        adj_mx = get_adjacency_matrix(args.graph_path, args.num_node, type='connectivity', id_filename=args.filename_id)
    elif args.graph_type == 'DISTANCE':
        adj_mx = get_Gaussian_matrix(args.graph_path, args.num_node, args.normalized_k, id_filename=args.filename_id)
    print("The shape of adjacency matrix : ", adj_mx.shape)
    adj_mx = torch.FloatTensor(adj_mx).to(args.device)
    return adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    # print the number of model parameters
    print_model_parameters(model, only_num=False)
    return model

def generate_model_components(args):
    init_seed(args.seed)
    # 1. model
    generator = Generator(args)
    generator = generator.to(args.device)
    generator = init_model(generator)

    discriminator = Discriminator(args)
    discriminator = discriminator.to(args.device)
    discriminator = init_model(discriminator)
    
    discriminator_rf = Discriminator_RF(args)
    discriminator_rf = discriminator_rf.to(args.device)
    discriminator_rf = init_model(discriminator_rf)

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
        loss_G = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss_G = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss_G = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss_G = torch.nn.SmoothL1Loss().to(args.device)
    elif args.loss_func == 'huber':
        loss = torch.nn.HuberLoss(delta=1.0).to(args.device)
    else:
        raise ValueError
    loss_D = torch.nn.BCELoss()
    # 3. optimizer
    optimizer_G = torch.optim.Adam(params=generator.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
    optimizer_D = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr_init * 0.1, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
    optimizer_D_RF = torch.optim.Adam(params=generator.parameters(), lr=args.lr_init * 0.1, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
    # 4. learning rate decay
    lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF = None, None, None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G,
                                                              milestones=lr_decay_steps,
                                                              gamma=args.lr_decay_rate)

        lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D,
                                                              milestones=lr_decay_steps,
                                                              gamma=args.lr_decay_rate)

        lr_scheduler_D_RF = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D_RF,
                                                                 milestones=lr_decay_steps,
                                                                 gamma=args.lr_decay_rate)
    return generator, discriminator, discriminator_rf, loss_G, loss_D, optimizer_G, optimizer_D, optimizer_D_RF, lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF


def get_log_dir(model, dataset, debug):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))  # /GNN4Traffic/TrafficFlow
    log_dir = os.path.join(current_dir,'log', model, dataset, current_time) 
    if os.path.isdir(log_dir) == False and not debug:
        os.makedirs(log_dir, exist_ok=True)  # run.log
    return log_dir


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'
    adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset, args.debug)
    generator, discriminator, discriminator_rf, loss_G, loss_D, optimizer_G, optimizer_D, optimizer_D_RF, lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF = generate_model_components(args)
    trainer = Trainer(
        args, 
        train_dataloader, 
        val_dataloader, 
        test_dataloader, 
        scaler, 
        generator, discriminator, discriminator_rf, 
        loss_G, loss_D, 
        optimizer_G, optimizer_D, optimizer_D_RF, 
        lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF
    )
    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/DAAGCN/PEMSD4/20230210112045/PEMSD4_DAAGCN_best_model.pth"
        trainer.test(generator, args, test_dataloader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError