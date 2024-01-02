import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn 
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, Informer, FEDformer, Autoformer, PatchTST, DLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.custom_loss import CLoss, HybridLoss
from torch.optim import lr_scheduler 
import csv

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer' : Informer,
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'PatchTST': PatchTST,
            'DLinear': DLinear
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
        
    def _select_criterion(self):
        if self.args.loss == 'MSELoss':
            criterion = nn.MSELoss()
            print('criterion: MSELoss')
            return criterion
        elif self.args.loss == 'HuberLoss':
            criterion = nn.HuberLoss(reduction='mean', delta=self.args.delta)
            print('criterion: HuberLoss')
            return criterion
        elif self.args.loss == 'L1Loss':
            criterion = nn.L1Loss()
            print('criterion: L1Loss')
            return criterion
        elif self.args.loss == 'CLoss':
            criterion = CLoss()
            print('criterion: CLoss')
            return criterion
        elif self.args.loss == 'HybridLoss':
            criterion = HybridLoss()
            print('criterion: HybridLoss')
            return criterion
        
    def _l1_regularizer(self, model, lambda_l1=0.01):
        lambda_l1 = self.args.lambda_l1
        lossl1 = 0
        for model_param_name, model_param_value in model.named_parameters():
                if model_param_name.endswith('weight'):
                    lossl1 += lambda_l1 * model_param_value.abs().sum()
        return lossl1    
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)              
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                
                outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        ##############################################################################
        train_loss_epochs = []
        vali_loss_epochs = []
        test_loss_epochs = []
        ##############################################################################
         
        exp_result_path =  './EXP_results/'
        if not os.path.exists(exp_result_path):
            os.makedirs(exp_result_path)
        path = f'{exp_result_path}{setting}/'
        if not os.path.exists(path):
            os.makedirs(path)
        check_point_path = f'{exp_result_path}{setting}/{self.args.checkpoints}/'
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        loss_path = f'{exp_result_path}{setting}/loss/'
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)
            
        with open(f'{exp_result_path}{setting}/hyper_parameters.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for arg in vars(self.args):
                value = getattr(self.args, arg)
                writer.writerow([arg, value]) 

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
               
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                    steps_per_epoch = train_steps,
                                    pct_start = self.args.pct_start,
                                    epochs = self.args.train_epochs,
                                    max_lr = self.args.learning_rate) 

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.regularization == True:              
                        loss = criterion(outputs, batch_y) + self._l1_regularizer(self.model)
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            #########################################################################################
            train_loss_epochs.append(train_loss)
            vali_loss_epochs.append(vali_loss)
            test_loss_epochs.append(test_loss)  
            #########################################################################################
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, check_point_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != 'TST':        
                #adjust_learning_rate(model_optim, epoch + 1, self.args)
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))    
                
        # best_model_path = path + '/' + 'checkpoint.pth'
        best_model_path = check_point_path + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        ########################################################### 
        train_loss_epochs = np.array(train_loss_epochs)
        vali_loss_epochs = np.array(vali_loss_epochs)
        test_loss_epochs = np.array(test_loss_epochs)
        np.save(loss_path + 'train_loss_epochs.npy', train_loss_epochs)
        np.save(loss_path + 'vali_loss_epochs.npy', vali_loss_epochs)
        np.save(loss_path + 'test_loss_epochs.npy', test_loss_epochs)
        ###########################################################
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []

        test_results_path = './EXP_results/' + setting + '/' + 'test_results/'
        if not os.path.exists(test_results_path):
            os.makedirs(test_results_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                
                outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                preds.append(pred)
                trues.append(true)

                inputx.append(batch_x.detach().cpu().numpy())
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(test_results_path, str(i) + '.pdf'))
                    
                    
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
      
        # result save
        
        folder_path = './EXP_results/' + setting + '/' + 'result/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        mae, mse, rmse, mape, mase, mspe  = metric(preds, trues)
        
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mase:{}'.format(mse, mae, rmse, mape, mase))
        f = open(f"./EXP_results/result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mase:{}'.format(mse, mae, rmse, mape, mase))
        f.write('\n')
        f.write('\n')
        f.close()
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, mase]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = './EXP_results/' + setting + self.args.checkpoints
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './EXP_results/' + setting + '/' + 'result/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return