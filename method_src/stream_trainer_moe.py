from logging import getLogger
import os
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F
from method_src.metrics import *

class StreamTrainerMoe(object):
    def __init__(self, args, pretrained_model, pretrained_classifier, pretrained_calibrator, device, train_dataloader, test_dataloader):
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

        self.model = copy.deepcopy(pretrained_model)
        self.model.to(self.device)
        self.pretrained_model = pretrained_model
        self.pretrained_model.to(self.device)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.classifier = pretrained_classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.calibrator = pretrained_calibrator
        for param in self.calibrator.parameters():
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
        all_first_pay_gmv_labels = []
        
        multi_tag = []
        metrics_dict = {}

        true_single = 0
        true_multi = 0
        total_single = 0
        total_multi = 0

        day_loader = self.test_dataloader.get_day_dataloader(day)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Testing Day {day + 1}", leave=False)
        self.logger.info(f"Testing Day {day + 1}...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_gmv_label = batch['final_gmv'].to(self.device)
                first_pay_gmv_label = batch['first_pay_gmv'].to(self.device)

                s_pgmv, m_pgmv = self.model(features)
                s_pgmv = s_pgmv.view(-1)
                m_pgmv = m_pgmv.view(-1)
                pretrain_s_pgmv, pretrain_m_pgmv = self.pretrained_model(features) 
                pretrain_s_pgmv = pretrain_s_pgmv.view(-1)
                pretrain_m_pgmv = pretrain_m_pgmv.view(-1)
                
                cls_logit = self.classifier(features)
                cls_prob = torch.sigmoid(cls_logit/0.5).view(-1) 

                pred_sigle = (cls_prob <= 0.1)
                pred_multi = (cls_prob >= 0.9)
                pred_mix = (cls_prob > 0.1) & (cls_prob < 0.9)

                pgmv = torch.zeros_like(pay_gmv_label)
                pgmv[pred_sigle] = s_pgmv[pred_sigle]
                pgmv[pred_multi] = m_pgmv[pred_multi]
                pgmv[pred_mix] = (1-cls_prob[pred_mix]) * s_pgmv[pred_mix] + cls_prob[pred_mix] * m_pgmv[pred_mix]
        
                pretrain_pgmv = torch.zeros_like(pay_gmv_label)
                pretrain_pgmv[pred_sigle] = pretrain_s_pgmv[pred_sigle]
                pretrain_pgmv[pred_multi] = pretrain_m_pgmv[pred_multi]
                pretrain_pgmv[pred_mix] = (1-cls_prob[pred_mix]) * pretrain_s_pgmv[pred_mix] + cls_prob[pred_mix] * pretrain_m_pgmv[pred_mix]
                
                true_single += (batch['multi_tag'][pred_sigle.cpu()] == 0).sum().item()
                true_multi += (batch['multi_tag'][pred_multi.cpu()] == 1).sum().item()
                total_single += pred_sigle.sum().item()
                total_multi += pred_multi.sum().item()

                all_gmv_preds.append(pgmv.view(-1).cpu())
                pretrain_all_gmv_preds.append(pretrain_pgmv.view(-1).cpu())
                all_gmv_labels.append(pay_gmv_label.cpu())
                multi_tag.append(batch['multi_tag'].cpu())
                all_first_pay_gmv_labels.append(first_pay_gmv_label.cpu())

        all_gmv_preds = torch.cat(all_gmv_preds, dim=0).numpy()
        pretrain_all_gmv_preds = torch.cat(pretrain_all_gmv_preds, dim=0).numpy()
        all_gmv_labels = torch.cat(all_gmv_labels, dim=0).numpy()
        all_first_pay_gmv_labels = torch.cat(all_first_pay_gmv_labels, dim=0).numpy()
        multi_tag = torch.cat(multi_tag, dim=0).numpy()
    
        test_acc = get_acc(all_gmv_preds, all_gmv_labels)
        test_auc = get_auc(all_gmv_preds, all_gmv_labels)
        test_alpr = get_alpr(all_gmv_preds, all_gmv_labels)

        pretrain_test_acc = get_acc(pretrain_all_gmv_preds, all_gmv_labels)
        pretrain_test_auc = get_auc(pretrain_all_gmv_preds, all_gmv_labels)
        pretrain_test_alpr = get_alpr(pretrain_all_gmv_preds, all_gmv_labels)

        test_result = f"Day {day + 1} - Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f}, Test ALPR: {test_alpr:.4f}\n"
        test_result += f"Pretrain Test AUC: {pretrain_test_auc:.4f}, Pretrain Test ACC: {pretrain_test_acc:.4f}, Pretrain ALPR:{pretrain_test_alpr:.4f}\n"
        self.logger.info(test_result)

        metrics_dict['test_auc'] = test_auc
        metrics_dict['test_acc'] = test_acc
        metrics_dict['test_alpr'] = test_alpr

        metrics_dict['pretrain_test_auc'] = pretrain_test_auc
        metrics_dict['pretrain_test_acc'] = pretrain_test_acc
        metrics_dict['pretrain_test_alpr'] = pretrain_test_alpr

        return metrics_dict


    def train(self):
        all_day_metrics = []

        for day in tqdm(range(len(self.train_dataloader)), desc="Days"):
            # Training
            for epoch_idx in tqdm(range(self.epochs), desc="Epochs", leave=False):
                self.model.train()
                total_loss = 0.0
                main_loss = 0.0
                ga_loss = 0.0
                day_loader = self.train_dataloader.get_day_dataloader(day)
                tqdm_day_dataloader = tqdm(day_loader, desc=f"Training Day {day} - Epoch {epoch_idx + 1}", leave=False)
                for batch_idx, batch in enumerate(tqdm_day_dataloader):
                    features = batch['features'].to(self.device)
                    pay_gmv_label = batch['first_pay_gmv'].to(self.device)
                    multi_tag = batch['multi_tag'].to(self.device)
                    t = ((batch['pay_ts'].float() - batch['click_ts'].float()) / (7*24*3600)).to(self.device)
                    now_count = batch['now_count'].to(self.device)

                    self.optimizer.zero_grad()
    
                    if "reader" in self.args.model:
                        prev_t = ((batch['prev_pay_ts'].float() - batch['click_ts'].float()) / (7*24*3600)).to(self.device)
                        prev_count = now_count - 1
                        prev_gmv_label = batch['prev_pay_gmv'].to(self.device)

                        s_pgmv, m_pgmv = self.model(features)
                        s_pgmv = s_pgmv.view(-1)
                        m_pgmv = m_pgmv.view(-1)
                        with torch.no_grad():
                            cls_logit = self.classifier(features)
                            cls_prob = torch.sigmoid(cls_logit/0.5).view(-1) # [batch_size, 1]
                            pho_preds = self.calibrator(features, t, now_count).view(-1)
    
                            mask_multi = (((now_count > 2) & (batch['pay_ts'] == batch['click_ts'] + 7 * 24 * 60 * 60).to(self.device)) | \
                            ((now_count > 1) & (batch['pay_ts'] < batch['click_ts'] + 7 * 24 * 60 * 60).to(self.device)))
                            cls_prob[mask_multi] = 1.0
                        
                            mask_single = ((batch['pay_ts'] == batch['click_ts'] + 7 * 24 * 60 * 60) & (batch['multi_tag'] == 0)).to(self.device)
                            cls_prob[mask_single] = 0.0

                            cls_logit_prev = self.classifier(features)
                            cls_prob_prev = torch.sigmoid(cls_logit_prev/0.5).view(-1) # [batch_size, 1]
                            pho_preds_prev = self.calibrator(features, prev_t, prev_count).view(-1)
                            mask_multi_prev = (prev_count > 1)
                            cls_prob_prev[mask_multi_prev] = 1.0

                        pred_sigle = (cls_prob <= 0.1)
                        pred_multi = (cls_prob >= 0.9)
                        pred_mix = (cls_prob > 0.1) & (cls_prob < 0.9)
                  

                        single_label = pay_gmv_label
                        fix_multi_label = torch.expm1(torch.log1p(pay_gmv_label) + pho_preds)
                        fix_mix_label = (1-cls_prob) * pay_gmv_label + cls_prob * torch.expm1(torch.log1p(pay_gmv_label) + pho_preds)
   
                        y_preds = torch.zeros_like(pay_gmv_label) # [batch_size]
                        y_preds[pred_sigle] = s_pgmv[pred_sigle]
                        y_preds[pred_multi] = m_pgmv[pred_multi]
                        y_preds[pred_mix] = (1-cls_prob[pred_mix]) * s_pgmv[pred_mix] + cls_prob[pred_mix] * m_pgmv[pred_mix]
                        
                        y_labels = torch.zeros_like(pay_gmv_label)
                        y_labels[pred_sigle] = single_label[pred_sigle]
                        y_labels[pred_multi] = fix_multi_label[pred_multi]
                        y_labels[pred_mix] = fix_mix_label[pred_mix]
                  
                        mask_attri_back = (batch['pay_ts'] == batch['click_ts'] + 7 * 24 * 60 * 60).to(self.device)
                        y_labels[mask_attri_back] = pay_gmv_label[mask_attri_back]

                        prev_single = (cls_prob_prev <= 0.1)
                        prev_multi = (cls_prob_prev >= 0.9)
                        prev_mix = (cls_prob_prev > 0.1) & (cls_prob_prev < 0.9)

                        prev_single_label = prev_gmv_label
                        prev_multi_label = torch.expm1(torch.log1p(prev_gmv_label) + pho_preds_prev)
                        prev_fix_label = (1-cls_prob_prev) * prev_gmv_label + cls_prob_prev * prev_multi_label
                        
                        prev_y_labels = torch.zeros_like(pay_gmv_label)
                        prev_y_labels[prev_single] = prev_single_label[prev_single]
                        prev_y_labels[prev_multi] = prev_multi_label[prev_multi]
                        prev_y_labels[prev_mix] = prev_fix_label[prev_mix]

                        ###########################################loss #################################################
                       
                        main_loss_each = torch.log1p(y_preds) - torch.log1p(y_labels) ###
                        main_loss = torch.abs(main_loss_each).mean() ###
                        
                        ga_loss_each = torch.log1p(y_preds) - torch.log1p(prev_y_labels)
                        now_loss_each = torch.log1p(y_preds) - torch.log1p(y_labels)
                        mask_first_prev = (batch['pay_ts'] == batch['click_ts'] + 7 * 24 * 60 * 60) 
                        ga_loss = torch.abs(ga_loss_each[mask_first_prev]).mean()
                        now_loss = torch.abs(now_loss_each[mask_first_prev]).mean()

                        loss = main_loss + self.args.boost_weight * (now_loss - self.args.ga_loss_weight * ga_loss) 

                    elif "vanilla" in self.args.model:
                        s_pgmv, m_pgmv = self.model(features)
                        s_pgmv = s_pgmv.view(-1)
                        m_pgmv = m_pgmv.view(-1)
                        with torch.no_grad():
                            cls_logit = self.classifier(features)
                            cls_prob = torch.sigmoid(cls_logit/0.5).view(-1) # [batch_size, 1]
                            pho_preds = self.calibrator(features, t, now_count).view(-1)
                            mask_multi = (now_count > 1)
                            cls_prob[mask_multi] = 1.0
                        pred_sigle = (cls_prob <= 0.1)
                        pred_multi = (cls_prob >= 0.9)
                        pred_mix = (cls_prob > 0.1) & (cls_prob < 0.9)
            
                        y_preds = torch.zeros_like(pay_gmv_label) 
                        y_preds[pred_sigle] = s_pgmv[pred_sigle]
                        y_preds[pred_multi] = m_pgmv[pred_multi]
                        y_preds[pred_mix] = (1-cls_prob[pred_mix]) * s_pgmv[pred_mix] + cls_prob[pred_mix] * m_pgmv[pred_mix]
                       
                        y_labels = pay_gmv_label

                        loss = torch.abs(torch.log1p(y_preds) - torch.log1p(y_labels)).mean()
                    else:
                        raise ValueError("traininig error! No such model: {}".format(self.args.model))   

                    
                    
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    
                self.lr_scheduler.step()
                avg_loss = total_loss / len(day_loader)
                self.logger.info(f"Day {day} - Epoch {epoch_idx + 1} - Avg Loss: {avg_loss:.4f}")
            
            # Test
            metrics_dict = self.test(day)
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
    
