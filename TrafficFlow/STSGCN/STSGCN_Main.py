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
from STSGCN_Config import args
from STSGCN_Trainer import Trainer
from STSGCN_Utils import *
from model.STSGCN.stsgcn import STSGCN as Network


def load_data(args):
    data = load_dataset(args, args.batch_size, args.batch_size, args.batch_size)
    train_dataloader, val_dataloader, test_dataloader, scaler = data['train_loader'], data['val_loader'], data['test_loader'], data['scaler']
    if args.graph_type == 'BINARY':
        adj_mx = get_adjacency_matrix(args.graph_path, args.num_nodes, type='connectivity', id_filename=args.filename_id)
    elif args.graph_type == 'DISTANCE':
        adj_mx = get_Gaussian_matrix(args.graph_path, args.num_nodes, args.normalized_k, id_filename=args.filename_id)
    adj_mx = construct_adj(adj_mx, steps=3)
    print("The shape of adjacency matrix : ", adj_mx.shape)
    adj_mx = torch.FloatTensor(adj_mx).to(args.device)
    return adj_mx, data, train_dataloader, val_dataloader, test_dataloader, scaler


def generate_model_components(args, adj_mx):
    # 1. model
    model = Network(
        adj=adj_mx,
        history=args.window, 
        num_of_vertices=args.num_nodes, 
        in_dim=args.input_dim, 
        hidden_dims=args.hidden_dims,
        first_layer_embedding_size=args.first_layer_embedding_size, 
        out_layer_dim=args.out_layer_dim, 
        activation=args.activation, 
        use_mask=args.use_mask,
        temporal_emb=args.temporal_emb, 
        spatial_emb=args.spatial_emb, 
        horizon=args.horizon, 
        strides=args.strides
    )
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p, gain=0.0003)
        else:
            nn.init.uniform_(p)
    # print the number of model parameters
    print_model_parameters(model, only_num=False)
    # 2. loss
    def masked_mae_loss(scaler, mask_value):
        def loss(preds, labels):
            if scaler:
                preds = scaler.inverse_transform(preds)
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
        loss = huber_loss
    else:
        raise ValueError
    # 3. optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init)
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
    adj_mx, data, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset, args.debug)
    model, loss, optimizer, lr_scheduler = generate_model_components(args, adj_mx)
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
        checkpoint = "../log/STSGCN/PEMSD4/20230213095104/PEMSD4_STSGCN_best_model.pth"
        trainer.test(model, args, data, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError