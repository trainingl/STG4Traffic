import sys
sys.path.append('../')

import os
import copy
import torch
import time
from tqdm import tqdm
from lib.utils import get_logger
from lib.evaluate import All_Metrics

class Trainer(object):
    def __init__(self, args, train_loader, val_loader, test_loader, scaler, 
    generator, discriminator, discriminator_rf, loss_G, loss_D, 
    optimizer_G, optimizer_D, optimizer_D_RF, lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF):
        super(Trainer, self).__init__()
        self.args = args
        self.train_loader = train_loader
        self.train_per_epoch = len(train_loader)
        self.val_loader = val_loader
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        else:
            self.val_loader = test_loader
            self.val_per_epoch = len(self.val_loader)
        self.test_loader = test_loader
        self.scaler = scaler
        # 模型、损失函数、优化器
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_rf = discriminator_rf
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_D_RF = optimizer_D_RF
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.lr_scheduler_D_RF = lr_scheduler_D_RF
        # 日志与模型的保存路径
        self.best_path = os.path.join(args.log_dir, '{}_{}_best_model.pth'.format(args.dataset, args.model))
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)  # run.log
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info("Experiment log path in: {}".format(args.log_dir))


    def train_epoch(self):
        self.generator.train()
        total_loss_G = 0
        total_loss_D = 0
        total_loss_D_RF = 0
        for _, (data, target) in enumerate(self.train_loader):
            batch_size = data.shape[0]
            data = data[..., :self.args.input_dim] # [B'', W, N, 1]
            label = target[..., :self.args.output_dim]  # # [B'', H, N, 1]

            # Adversarial ground truths
            cuda = True if torch.cuda.is_available() else False
            TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            valid = torch.autograd.Variable(TensorFloat(batch_size*(self.args.window + self.args.horizon), 1).fill_(1.0), requires_grad=False)
            fake = torch.autograd.Variable(TensorFloat(batch_size*(self.args.window + self.args.horizon), 1).fill_(0.0), requires_grad=False)

            valid_rf = torch.autograd.Variable(TensorFloat(batch_size*self.args.num_node, 1).fill_(1.0), requires_grad=False)
            fake_rf = torch.autograd.Variable(TensorFloat(batch_size*self.args.num_node, 1).fill_(0.0), requires_grad=False)

            #-------------------------------------------------------------------
            # Train Generator 
            #-------------------------------------------------------------------
            self.optimizer_G.zero_grad()
                        
            # data and target shape: B, W, N, F, and B, H, N, F; output shape: B, H, N, F (F=1)
            output = self.generator(data)
            if self.args.real_value: # it is depended on the output of model. If output is real data, the label should be reversed to real data
                label = self.scaler.inverse_transform(label)
            
            fake_input = torch.cat((data, self.scaler.transform(output)), dim=1) if self.args.real_value else torch.cat((data, output), dim=1) # [B'', W, N, 1] // [B'', H, N, 1] -> [B'', W+H, N, 1]
            true_input = torch.cat((data, self.scaler.transform(label)), dim=1) if self.args.real_value else torch.cat((data, label), dim=1)

            fake_input_rf = self.scaler.transform(output) if self.args.real_value else output # [B'', W, N, 1] // [B'', H, N, 1] -> [B'', W+H, N, 1]
            true_input_rf = self.scaler.transform(label) if self.args.real_value else label
            
            loss_G = self.loss_G(output.cuda(), label) + 0.01 * self.loss_D(self.discriminator(fake_input), valid) + self.loss_D(self.discriminator_rf(fake_input_rf), valid_rf)
            loss_G.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)
            self.optimizer_G.step()
            total_loss_G += loss_G.item()

            #-------------------------------------------------------------------
            # Train Discriminator 
            #-------------------------------------------------------------------
            self.optimizer_D.zero_grad()
            real_loss = self.loss_D(self.discriminator(true_input), valid)
            fake_loss = self.loss_D(self.discriminator(fake_input.detach()), fake)
            loss_D = 0.5 * (real_loss + fake_loss)
            loss_D.backward()
            self.optimizer_D.step() 
            total_loss_D += loss_D.item()

            #-------------------------------------------------------------------
            # Train Discriminator_RF
            #-------------------------------------------------------------------
            self.optimizer_D_RF.zero_grad()
            real_loss_rf = self.loss_D(self.discriminator_rf(true_input_rf), valid_rf)
            fake_loss_rf = self.loss_D(self.discriminator_rf(fake_input_rf.detach()), fake_rf)
            loss_D_RF = 0.5 * (real_loss_rf + fake_loss_rf)
            loss_D_RF.backward()
            self.optimizer_D_RF.step() 
            total_loss_D_RF += loss_D_RF.item()
        
        train_epoch_loss_G = total_loss_G / self.train_per_epoch # average generator loss
        train_epoch_loss_D = total_loss_D / self.train_per_epoch # average discriminator loss
        train_epoch_loss_D_RF = total_loss_D_RF / self.train_per_epoch # average discriminator loss
        # learning rate decay
        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            self.lr_scheduler_D_RF.step()
        return train_epoch_loss_G, train_epoch_loss_D, train_epoch_loss_D_RF

    def val_epoch(self):
        self.generator.eval()
        total_val_loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.val_loader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.generator(data)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)  # 若模型预测的真实值
                loss = self.loss_G(output.cuda(), label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        return val_loss


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            t1 = time.time()
            train_epoch_loss_G, _, _ = self.train_epoch()
            t2 = time.time()
            # 验证, 如果是Encoder-Decoder结构，则需要将epoch作为参数传入
            val_epoch_loss = self.val_epoch()
            t3 = time.time()
            self.logger.info('Epoch {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f} secs, Inference Time: {:.4f} secs.'.format(epoch, train_epoch_loss_G, val_epoch_loss, (t2 - t1), (t3 - t2)))
            train_loss_list.append(train_epoch_loss_G)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss_G > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # is or not early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info("Current best model saved!")
                best_model = copy.deepcopy(self.generator.state_dict())
                torch.save(best_model, self.best_path)
            
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f} min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # save the best model to file
        self.logger.info("Saving current best model to " + self.best_path)
        # load model and test
        self.generator.load_state_dict(best_model)
        self.test(self.generator, self.args, self.test_loader, self.scaler, self.logger)


    @staticmethod
    def test(model, args, data_loader, scaler, logger, save_path=None):
        if save_path != None:
            model.load_state_dict(torch.load(save_path))
            model.to(args.device)
            print("load saved model...")
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(data)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:   # 预测值是真实值还是归一化的结果
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))