from logging import getLogger
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from method_src.metrics import *
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

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
        self.loss_pho_weight = args.loss_pho_weight

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
        all_pho_preds = []
        all_pho_labels = []
        all_first_pay_gmv_labels = []
        multi_tag = []
        all_prob_preds = []
        all_logit_preds = []

        metrics_dict ={}
        self.logger.info("Testing...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                features = batch['features'].to(self.device)
                pay_gmv_label = batch['final_gmv'].to(self.device)
                pho_label = (torch.log1p(batch['final_gmv']) - torch.log1p(batch['first_pay_gmv'])).to(self.device)
                first_pay_gmv_label = batch['first_pay_gmv'].to(self.device)
                multi_tag_label = batch['multi_tag'].to(self.device)
                mask_single_purchase = (multi_tag_label == 0)
                t = ((batch['pay_ts'].float() - batch['click_ts'].float()) / (7*24*3600)).to(self.device)
                if 'calibrator' in self.args.model:
                    now_count = batch['now_count'].to(self.device)
        
     
                if self.args.model == 'classifier':
                    logit = self.model(features)
                    prob = torch.sigmoid(logit/0.5)
                elif self.args.model == 'calibrator':
                    pho_preds = self.model(features, t, now_count)
                else:
                    raise ValueError(f"Unsupported model for testing: {self.args.model}")

                if self.args.model == 'classifier':
                    all_prob_preds.append(prob.view(-1).cpu())
                    multi_tag.append(batch['multi_tag'].cpu())
                    all_logit_preds.append(logit.view(-1).cpu())
                elif self.args.model == 'calibrator':
                    all_pho_preds.append(pho_preds.view(-1).cpu())
                    all_pho_labels.append(pho_label.cpu())
                    all_first_pay_gmv_labels.append(first_pay_gmv_label.cpu())
                    all_gmv_labels.append(pay_gmv_label.cpu())
                else:
                    raise ValueError(f"Unsupported model for testing: {self.args.model}")

        if self.args.model == 'classifier':
            all_logit_preds = torch.cat(all_logit_preds, dim=0).numpy()
            all_prob_preds = torch.cat(all_prob_preds, dim=0).numpy()
            multi_tag = torch.cat(multi_tag, dim=0).numpy()
            class_preds = (all_prob_preds >= 0.5).astype(int)
        elif self.args.model == 'calibrator':
            all_pho_preds = torch.cat(all_pho_preds, dim=0).numpy()
            all_pho_labels = torch.cat(all_pho_labels, dim=0).numpy()
            all_first_pay_gmv_labels = torch.cat(all_first_pay_gmv_labels, dim=0).numpy()
            all_gmv_labels = torch.cat(all_gmv_labels, dim=0).numpy()
        else:
            raise ValueError(f"Unsupported model for testing: {self.args.model}")


        if self.args.model == 'classifier':
            self.logger.info('===================classification preds===================')
            all_auc = roc_auc_score(multi_tag, all_prob_preds)
            all_acc = accuracy_score(multi_tag, class_preds)
            all_precision = precision_score(multi_tag, class_preds)
            all_recall = recall_score(multi_tag, class_preds)
            self.logger.info(f'ALL auc: {all_auc}, acc: {all_acc}, precision: {all_precision}, recall: {all_recall}')

        if self.args.model == 'calibrator':
            self.logger.info('===================pho preds===================')
            predict_gmv = np.expm1(np.log1p(all_first_pay_gmv_labels) + all_pho_preds)

            pho_mae = get_mae(all_pho_preds, all_pho_labels)
            pho_predict_mape = get_mape(predict_gmv, all_gmv_labels)
            pho_predict_pcoc = get_pcoc(predict_gmv, all_gmv_labels)
            pho_predict_acc = get_acc(predict_gmv, all_gmv_labels)
            pho_predict_alpr = get_alpr(predict_gmv, all_gmv_labels)
            self.logger.info(f'pho mae: {pho_mae:.4f}, pho predict mape: {pho_predict_mape:.4f}, pho predict pcoc: {pho_predict_pcoc:.4f}, pho predict acc: {pho_predict_acc:.4f} , pho predict alpr: {pho_predict_alpr:.4f}')

        return metrics_dict

    def train(self, train_loader, test_loader):
        if os.path.isfile(self.model_pth):
            self.logger.info("model_pth {} exists.".format(self.model_pth))
            self.logger.info("loading pretrain model...")
            self.model.load_state_dict(torch.load(self.model_pth, map_location=self.device))
            metrics_dict = self.test(test_loader)
        else:
            for epoch_idx in range(self.epochs):
                self.model.train()
                tqdm_dataloder = tqdm(train_loader)
                total_loss = 0.0
                cls_total_loss = 0.0

                cls_auc = 0.0
                cls_acc = 0.0
                cls_recall = 0.0
                cls_precision = 0.0
                
                self.logger.info(f"Epoch {epoch_idx+1}/{self.epochs} training...")
                for batch_idx, batch in enumerate(tqdm_dataloder):
                    features = batch['features'].to(self.device)
                    pay_gmv_label = batch['final_gmv'].to(self.device)
                    pay_first_gmv_label = batch['first_pay_gmv'].to(self.device)
                    multi_tag_label = batch['multi_tag'].to(self.device)
                    mask_single_purchase = (multi_tag_label == 0)
                    t = ((batch['pay_ts'].float() - batch['click_ts'].float()) / (7*24*3600)).to(self.device)
                    if 'calibrator' in self.args.model:
                        now_count = batch['now_count'].to(self.device)

                    self.optimizer.zero_grad()
                    
                    if self.args.model == 'classifier':
                        logit = self.model(features)
                        cls_loss = F.binary_cross_entropy_with_logits(logit.view(-1), multi_tag_label.float())
                        loss = cls_loss
                    elif self.args.model == 'calibrator':
                        pho_preds = self.model(features, t, now_count)
                        pho_true = torch.log1p(pay_gmv_label) - torch.log1p(pay_first_gmv_label)
                        loss_pho = F.l1_loss(pho_preds.view(-1), pho_true.view(-1))
                        loss = loss_pho
                    else:
                        raise ValueError(f"Unsupported model: {self.args.model}")

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item() 

                    if self.args.model == 'classifier':
                        prob = torch.sigmoid(logit)
                        all_prob = prob.view(-1).cpu().detach().numpy()
                        all_label = multi_tag_label.cpu().detach().numpy()
                        all_class_preds = (all_prob >= 0.5).astype(int)
                        cls_auc += roc_auc_score(all_label, all_prob)
                        cls_acc += accuracy_score(all_label, all_class_preds)
                        cls_recall += recall_score(all_label, all_class_preds)
                        cls_precision += precision_score(all_label, all_class_preds)
                        cls_total_loss += cls_loss.item()
                        tqdm_dataloder.set_description(f"Epoch {epoch_idx+1}/{self.epochs},  Cls Loss: {cls_total_loss/(batch_idx+1):.4f},  Cls AUC: {cls_auc/(batch_idx+1):.4f}, Cls Acc: {cls_acc/(batch_idx+1):.4f}, Cls Precision: {cls_precision/(batch_idx+1):.4f}, Cls Recall: {cls_recall/(batch_idx+1):.4f}")
                    elif self.args.model == 'calibrator':
                        tqdm_dataloder.set_description(f"Epoch {epoch_idx+1}/{self.epochs}, Loss: {total_loss/(batch_idx+1):.4f}")
                    else:
                        raise ValueError(f"Unsupported model: {self.args.model}")
                        
                self.lr_scheduler.step()

                parent_dir = os.path.dirname(self.model_pth)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                torch.save(self.model.state_dict(), self.model_pth)
                self.logger.info(f"Model saved at Epoch {epoch_idx+1}")

                