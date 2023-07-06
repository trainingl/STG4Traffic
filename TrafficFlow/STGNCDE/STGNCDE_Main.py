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
from STGNCDE_Utils import *
from STGNCDE_Config import args
from STGNCDE_Trainer import Trainer
from model.STGNCDE.GCDE import *
from model.STGNCDE.vector_fields import *

def load_data(args):
    train_dataloader, val_dataloader, test_dataloader, scaler, times = get_dataloader_cde(args, 
                                                                            normalizer=args.normalizer, 
                                                                            tod=False,
                                                                            dow=False, 
                                                                            single=False)
    if args.graph_type == 'BINARY':
        adj_mx = get_adjacency_matrix(args.graph_path, args.num_nodes, type='connectivity', id_filename=args.filename_id)
    elif args.graph_type == 'DISTANCE':
        adj_mx = get_Gaussian_matrix(args.graph_path, args.num_nodes, args.normalized_k, id_filename=args.filename_id)
    print("The shape of adjacency matrix : ", adj_mx.shape)
    adj_mx = torch.FloatTensor(adj_mx).to(args.device)
    return adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler, times


def generate_model_components(args):
    # 1. model
    vector_field_f = FinalTanh_f(
        input_channels=args.input_dim, 
        hidden_channels=args.hid_dim,
        hidden_hidden_channels=args.hid_hid_dim,
        num_hidden_layers=args.num_layers
    )
    vector_field_g = VectorField_g(
        input_channels=args.input_dim, 
        hidden_channels=args.hid_dim,
        hidden_hidden_channels=args.hid_hid_dim,
        num_hidden_layers=args.num_layers, 
        num_nodes=args.num_nodes, 
        cheb_k=args.cheb_order, 
        embed_dim=args.embed_dim,
        g_type=args.g_type
    )
    model = NeuralGCDE(
        args, 
        func_f=vector_field_f, 
        func_g=vector_field_g, 
        input_channels=args.input_dim, 
        hidden_channels=args.hid_dim,
        output_channels=args.output_dim, 
        initial=True, 
        device=args.device, 
        atol=1e-9, 
        rtol=1e-7, 
        solver=args.solver
    )
    model = model.to(args.device)
    if args.model_type == 'type1_temporal':
        vector_field_f = vector_field_f.to(args.device)
        vector_field_g = None
    elif args.model_type == 'type1_spatial':
        vector_field_f = None
        vector_field_g = vector_field_g.to(args.device)
    else:
        vector_field_f = vector_field_f.to(args.device)
        vector_field_g = vector_field_g.to(args.device)
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
    adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler, times = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset, args.debug)
    model, loss, optimizer, lr_scheduler = generate_model_components(args)
    trainer = Trainer(
        args=args, 
        train_loader=train_dataloader, 
        val_loader=val_dataloader, 
        test_loader=test_dataloader, 
        scaler=scaler, 
        times=times,
        model=model, 
        loss=loss, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler
    )
    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/STGNCDE/PEMSD8/20230108002915/PEMSD8_STGNCDE_best_model.pth"
        trainer.test(model, args, test_dataloader, scaler, trainer.logger, times, save_path=checkpoint)
    else:
        raise ValueError