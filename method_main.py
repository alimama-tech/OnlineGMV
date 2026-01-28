import argparse
import torch
from logging import getLogger
from method_src.pretrain import run
from method_src.online_stream import run_stream
from method_src.utils import *
import time
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./data/', help='Data path')
    parser.add_argument('--data_cache_path', type=str, default='./data/', help='Data cache path')
    parser.add_argument('--dataset_name', type=str, default='trace.txt', help='Dataset name')
    parser.add_argument('--gmv_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--mode', type=str, default="stream", choices=["pretrain", "stream"], help='training mode')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
  
    parser.add_argument('--model', type=str, default='pretrain_bdl', help='Model name')
    parser.add_argument('--pretrain_model', type=str, default='gmvcalibmlp', help='Pretrained Model Name')
    parser.add_argument('--embed_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization strength')
    parser.add_argument('--loss_pho_weight', type=float, default=1, help='pho loss weight')
 
    parser.add_argument('--gpu_id', type=str, default='3', help='Device index')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--stopping_step', type=int, default=2, help='Early stopping step')
    parser.add_argument('--learning_rate_scheduler', type=list, default=[0.95, 2], help='Learning rate scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--model_save_pth', type=str, default='./pretrain_model/', help='Model save pth')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--train_start_day', type=float, default=57, help='Training start day') 
    parser.add_argument('--train_end_day', type=float, default=81.875, help='Training end day') 
    parser.add_argument('--test_start_day', type=float, default=57.125, help='Test start day') 
    parser.add_argument('--test_end_day', type=float, default=82, help='Test end day') 
    parser.add_argument('--stream_step', type=float, default=0.125, help='how long to train') 
    parser.add_argument('--bdl_stream_step', type=float, default=1, help='pretrain bdl how long to train')  
    parser.add_argument('--ga_loss_weight', type=float, default=0, help='ga loss weight')
    parser.add_argument('--boost_weight', type=float, default=0, help='ga loss weight')
    
    parser.add_argument('--remarks', type=str, default='rerun', help='logger remark')
    parser.add_argument('--pkl_remarks', type=str, default='None', help='load stream pkl remark')
    parser.add_argument('--pretrain_remarks', type=str, default='None', help='load pretrain pkl remark')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    init_logger(args)
    logger = getLogger()

    all_res = ''
    init_seed(args.seed)

    logger.info(args)
    logger.info(f"===============lr_{args.lr} -bw_weight_{args.boost_weight} -ga_weight_{args.ga_loss_weight}================")
    if args.mode == "pretrain":
        run(args, device)
    else:
        res = run_stream(args, device)
        all_res += f"=============lr_{args.lr} -bw_weight_{args.boost_weight} - ga_weight_{args.ga_loss_weight}================\n{res}\n"
    logger.info(all_res)