from logging import getLogger
import os
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F
from src.metrics import *
import pickle

class StreamTrainer(object):
    def __init__(self, args, pretrained_model, pretrained_classifier, device, train_dataloader, test_dataloader):
        self.args = args
        self.device = device
        self.logger = getLogger()
        self.epochs = args.epochs

        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.stopping_step = args.stopping_step 
        self.learning_rate_scheduler = args.learning_rate_scheduler

        self.model_name = args.model
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.model_name}_{self.mode}.pth")

        self.model = copy.deepcopy(pretrained_model)
        self.model.to(self.device)
        self.pretrained_model = pretrained_model
        self.pretrained_model.to(self.device)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        if "twotower" in args.model:
            self.classifier = pretrained_classifier
            for param in self.classifier.parameters():
                param.requires_grad = False

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

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

    
    def test(self, day):
        self.model.eval()
        self.pretrained_model.eval()

        all_gmv_preds = []
        pretrain_all_gmv_preds = []
        all_gmv_labels = []
        multi_tag = []

        metrics_dict = {}

        day_loader = self.test_dataloader.get_day_dataloader(day)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Testing Day {day + 1}", leave=False)
        self.logger.info(f"Testing Day {day + 1}...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_gmv_label = batch['final_gmv'].to(self.device)
                multi_tag_label = batch['multi_tag'].to(self.device)

                if 'twotower' in self.args.model:
                    s_pgmv, m_pgmv = self.model(features)
                    pretrain_s_pgmv, pretrain_m_pgmv = self.pretrained_model(features)
                    mask_single_purchase = (multi_tag_label == 0)
                    pgmv = torch.zeros_like(pay_gmv_label)
                    pretrain_pgmv = torch.zeros_like(pay_gmv_label)
                    pgmv[mask_single_purchase] = s_pgmv[mask_single_purchase].view(-1)
                    pgmv[~mask_single_purchase] = m_pgmv[~mask_single_purchase].view(-1)


                    cls_logit = self.classifier(features)
                    cls_prob = torch.sigmoid(cls_logit/0.5).view(-1) 

                    pred_sigle = (cls_prob <= 0.1)
                    pred_multi = (cls_prob >= 0.9)
                    pred_mix = (cls_prob > 0.1) & (cls_prob < 0.9)

                    pretrain_pgmv = torch.zeros_like(pay_gmv_label)
                    pretrain_pgmv[pred_sigle] = pretrain_s_pgmv[pred_sigle].view(-1)
                    pretrain_pgmv[pred_multi] = pretrain_m_pgmv[pred_multi].view(-1)
                    pretrain_pgmv[pred_mix] = (1-cls_prob[pred_mix]) * pretrain_s_pgmv[pred_mix].view(-1) + cls_prob[pred_mix] * pretrain_m_pgmv[pred_mix].view(-1)
                        
                else:
                    pgmv = self.model(features)
                    pretrain_pgmv = self.pretrained_model(features)
                    
                all_gmv_preds.append(pgmv.view(-1).cpu())
                pretrain_all_gmv_preds.append(pretrain_pgmv.view(-1).cpu())
                all_gmv_labels.append(pay_gmv_label.cpu())
                multi_tag.append(batch['multi_tag'].cpu())

        all_gmv_preds = torch.cat(all_gmv_preds, dim=0).numpy()
        pretrain_all_gmv_preds = torch.cat(pretrain_all_gmv_preds, dim=0).numpy()
        all_gmv_labels = torch.cat(all_gmv_labels, dim=0).numpy()
        multi_tag = torch.cat(multi_tag, dim=0).numpy()


        test_acc = get_acc(all_gmv_preds, all_gmv_labels)
        test_auc = get_auc(all_gmv_preds, all_gmv_labels)
        test_alpr = get_alpr(all_gmv_preds, all_gmv_labels)
        
        pretrain_test_acc = get_acc(pretrain_all_gmv_preds, all_gmv_labels)
        pretrain_test_auc = get_auc(pretrain_all_gmv_preds, all_gmv_labels)
        pretrain_test_alpr = get_alpr(pretrain_all_gmv_preds, all_gmv_labels)

        test_result = f"Day {day + 1} - Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f},Test ALPR: {test_alpr:.4f}\n"
        test_result += f"Pretrain Test AUC: {pretrain_test_auc:.4f}, Pretrain Test ACC: {pretrain_test_acc:.4f}, Pretrain Test ALPR: {pretrain_test_alpr:.4f}\n"
        self.logger.info(test_result)

        metrics_dict['test_auc'] = test_auc
        metrics_dict['test_acc'] = test_acc
        metrics_dict['test_alpr'] = test_alpr
        metrics_dict['pretrain_test_auc'] = pretrain_test_auc
        metrics_dict['pretrain_test_acc'] = pretrain_test_acc
        metrics_dict['pretrain_test_alpr'] = pretrain_test_alpr

        return metrics_dict, multi_tag


    def train(self):
        all_day_metrics = []

        for day in tqdm(range(len(self.train_dataloader)), desc="Days"):
            # Training
            for epoch_idx in tqdm(range(self.epochs), desc="Epochs", leave=False):
                self.model.train()
                total_loss = 0.0
                day_loader = self.train_dataloader.get_day_dataloader(day)
                tqdm_day_dataloader = tqdm(day_loader, desc=f"Training Day {day} - Epoch {epoch_idx + 1}", leave=False)
                for batch_idx, batch in enumerate(tqdm_day_dataloader):
                    features = batch['features'].to(self.device)

                    if "oracle" in self.args.model: 
                        pay_gmv_label = batch['final_gmv'].to(self.device)
                    elif "vanilla" in self.args.model:
                        pay_gmv_label = batch['first_pay_gmv'].to(self.device)
                    else:
                        raise ValueError("Invalid model type")

                    self.optimizer.zero_grad()
                    
                    if 'twotower' in self.args.model:
                        multi_tag = batch['multi_tag'].to(self.device)  
                        mask_single_purchase = (multi_tag == 0)
                        s_pgmv, m_pgmv = self.model(features)
                        s_loss = torch.abs(torch.log1p(s_pgmv[mask_single_purchase].view(-1)) - torch.log1p(pay_gmv_label[mask_single_purchase])).mean()
                        m_loss = torch.abs(torch.log1p(m_pgmv[~mask_single_purchase].view(-1)) - torch.log1p(pay_gmv_label[~mask_single_purchase])).mean()
                        loss = s_loss + m_loss
                    else:
                        pamt = self.model(features)
                        loss = torch.abs(torch.log1p(pamt.view(-1)) - torch.log1p(pay_gmv_label)).mean()
                    
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    tqdm_day_dataloader.set_description(f"Epoch {epoch_idx+1}/{self.epochs}, Loss: {total_loss/(batch_idx+1):.4f}")
                    
                self.lr_scheduler.step()
                avg_loss = total_loss / len(day_loader)

                self.logger.info(f"Day {day} - Epoch {epoch_idx + 1} - Avg Loss: {avg_loss:.4f}")

            metrics_dict, multi_tag = self.test(day)
            all_day_metrics.append(metrics_dict)

        res = f'============lr: {self.args.lr}============\n'
        self.logger.info("Training completed for all days.")
        avg_metrics = self.aggregate_metrics(all_day_metrics)
        for k, v in avg_metrics.items():
            self.logger.info(f"Average {k}: {v:.4f}")
            res += f"Average {k}: {v:.4f}\n"
        res += '========================================\n'
        return res
               
    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total
    
