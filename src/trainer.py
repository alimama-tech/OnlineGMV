from logging import getLogger
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.metrics import *

class PretrainedTrainer(object):
    def __init__(self, args, model, device):
        self.args = args
        self.logger = getLogger()
        
        self.device = device
        self.epochs = args.epochs

        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.stopping_step = args.stopping_step 
        self.learning_rate_scheduler = args.learning_rate_scheduler

        self.model_name = args.model
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.model_name}_{self.mode}_{args.seed}_{args.pretrain_remarks}.pth")

        self.model = model.to(self.device)
        self.cur_step = 0
        self.best_epoch = 0
        self.best_pcoc = 0.0
        self.best_result = ''
        
        self.optimizer = self._get_optimizer()
        fac = lambda epoch: self.learning_rate_scheduler[0] ** (epoch / self.learning_rate_scheduler[1])
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

    def _get_optimizer(self):
        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        return optimizer

    def test(self, test_loader):
        self.model.eval()
        all_gmv_labels = []
        all_pamt_preds = []
        multi_tag = []

        metrics_dict ={}
        self.logger.info("Testing...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                features = batch['features'].to(self.device)
                pay_gmv_label = batch['final_gmv'].to(self.device)
                multi_tag_label = batch['multi_tag'].to(self.device)
    
                if 'twotower' in self.args.model:
                    s_pgmv, m_pgmv = self.model(features)
                    mask_single_purchase = (multi_tag_label == 0)

                    pamt = torch.zeros_like(pay_gmv_label)
                    pamt[mask_single_purchase] = s_pgmv[mask_single_purchase].view(-1)
                    pamt[~mask_single_purchase] = m_pgmv[~mask_single_purchase].view(-1)
                else:
                    pamt = self.model(features)

                all_pamt_preds.append(pamt.view(-1).cpu())
                all_gmv_labels.append(pay_gmv_label.cpu())
                multi_tag.append(batch['multi_tag'].cpu())

        all_pamt_preds = torch.cat(all_pamt_preds, dim=0)
        all_gmv_labels = torch.cat(all_gmv_labels, dim=0)

        all_pgmv_preds = all_pamt_preds

        all_pgmv_preds = all_pgmv_preds.numpy()
        all_gmv_labels = all_gmv_labels.numpy()
        multi_tag = torch.cat(multi_tag, dim=0).numpy()



        test_alpr = get_alpr(all_pgmv_preds, all_gmv_labels)
        self.logger.info('Test ALPR: {:.4f}'.format(test_alpr))
        test_acc = get_acc(all_pgmv_preds, all_gmv_labels)
        self.logger.info('Test ACC: {:.4f}'.format(test_acc))
        test_auc = get_auc(all_pgmv_preds, all_gmv_labels)
        self.logger.info('Test AUC: {:.4f}'.format(test_auc))

        test_result = f"Test ALL AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, ALPR: {test_alpr:.4f}"
        self.logger.info(f"{test_result}")

        metrics_dict['test_auc'] = test_auc
        metrics_dict['test_alpr'] = test_alpr
        metrics_dict['test_acc'] = test_acc

        test_result = f"Test ALL AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, ALPR: {test_alpr:.4f}"
        self.logger.info(f"{test_result}")

        return metrics_dict

    def train(self, train_loader, test_loader):
        if os.path.isfile(self.model_pth):
            self.logger.info("model_pth {} exists.".format(self.model_pth))
            self.logger.info("loading pretrain model...")
            self.model.load_state_dict(torch.load(self.model_pth, map_location=self.device))
        else:
            for epoch_idx in range(self.epochs):
                self.model.train()
                tqdm_dataloder = tqdm(train_loader)
                total_loss = 0.0
                total_s_loss = 0.0
                total_m_loss = 0.0
                self.logger.info(f"Epoch {epoch_idx+1}/{self.epochs} training...")
                for batch_idx, batch in enumerate(tqdm_dataloder):
                    features = batch['features'].to(self.device)
                    pay_gmv_label = batch['final_gmv'].to(self.device)
                    multi_tag = batch['multi_tag'].to(self.device)
                    mask_single_purchase = (multi_tag == 0)

                    self.optimizer.zero_grad()

                    if 'twotower' in self.args.model:
                        s_pgmv, m_pgmv = self.model(features)
                        s_loss = torch.abs(torch.log1p(s_pgmv[mask_single_purchase].view(-1)) - torch.log1p(pay_gmv_label[mask_single_purchase])).mean()
                        m_loss = torch.abs(torch.log1p(m_pgmv[~mask_single_purchase].view(-1)) - torch.log1p(pay_gmv_label[~mask_single_purchase])).mean()
                        loss = s_loss + m_loss
                        
                    else:
                        pamt = self.model(features)
                        loss = torch.abs(torch.log1p(pamt.view(-1)) - torch.log1p(pay_gmv_label)).mean() # 改成log训练,MAE

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item() 
                    total_s_loss += s_loss.item() if 'twotower' in self.args.model else 0.0
                    total_m_loss += m_loss.item() if 'twotower' in self.args.model else 0.0

                    if 'twotower' in self.args.model:
                        tqdm_dataloder.set_description(f"Epoch {epoch_idx+1}/{self.epochs}, Loss: {total_loss/(batch_idx+1):.4f}, S_Loss: {total_s_loss/(batch_idx+1):.4f}, M_Loss: {total_m_loss/(batch_idx+1):.4f}")
                    else:
                        tqdm_dataloder.set_description(f"Epoch {epoch_idx+1}/{self.epochs}, Loss: {total_loss/(batch_idx+1):.4f}")


                self.lr_scheduler.step()
                # training info
                avg_loss = total_loss / len(train_loader)
                self.logger.info(f"Epoch {epoch_idx+1}/{self.epochs}, Average Loss: {avg_loss:.4f}")

                parent_dir = os.path.dirname(self.model_pth)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                torch.save(self.model.state_dict(), self.model_pth)
                self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
            
        
            

                