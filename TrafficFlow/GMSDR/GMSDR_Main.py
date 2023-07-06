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
from GMSDR_Config import args
from GMSDR_Trainer import Trainer
from model.GMSDR.gmsdr import GMSDR as Network


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
    adj_mx = torch.FloatTensor(adj_mx)
    return adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler


def generate_model_components(args, adj_mx):
    # 1. model
    model = Network(
        adj_mx=adj_mx,
        input_dim = args.input_dim,
        output_dim = args.output_dim,
        rnn_units = args.rnn_units,
        seq_len = args.window,
        horizon = args.horizon,
        pre_k = args.pre_k,
        pre_v = args.pre_v,
        num_rnn_layers = args.num_rnn_layers,
        num_nodes = args.num_node,
        filter_type = args.filter_type,
        cl_decay_steps = args.cl_decay_steps,
        max_diffusion_step = args.max_diffusion_step,
        use_curriculum_learning = args.use_curriculum_learning
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
    def masked_mae_loss(preds, labels):
        return MAE_torch(preds, labels, mask_value=0.0)
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss
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
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-3,
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
    adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset, args.debug)
    model, loss, optimizer, lr_scheduler = generate_model_components(args, adj_mx)
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
        checkpoint = "../log/GMSDR/PEMSD4/20230212141206/PEMSD4_GMSDR_best_model.pth"
        trainer.test(model, args, test_dataloader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError