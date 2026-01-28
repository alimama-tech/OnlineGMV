import os
import torch
from logging import getLogger
from src.data import get_stream_dataset
from src.dataset import StreamDataLoader
from src.models.BaselineMLP import BaselineMLP
from src.stream_trainer import StreamTrainer
from src.models.TwoTower import SharedBottomTwoTower
from src.models.Classifier import BaseMLP

logger = getLogger()

def run_stream(args,device):
    # data
    train_stream, test_stream = get_stream_dataset(args)
    train_stream_dataloader = StreamDataLoader(train_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_stream_dataloader = StreamDataLoader(test_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
 
    if 'twotower' in args.model:
        pretrained_model = SharedBottomTwoTower(args).to(device)
        pretrained_model.load_state_dict(torch.load(os.path.join(args.model_save_pth, f"sharebottom_twotower_pretrain_{args.seed}_{args.pretrain_remarks}.pth"),map_location=device))
        logger.info(f"pretrained_model: sharedbottom_twotower_pretrain_{args.seed}_{args.pretrain_remarks}.pth")
        pretrained_classifier = BaseMLP(args).to(device)
        pretrained_classifier.load_state_dict(torch.load(os.path.join(args.model_save_pth, f"classifier_pretrain_{args.seed}_classifier.pth"),map_location=device))
        logger.info(f"Pretrained Classifier:classifier_pretrain_{args.seed}_classifier.pth")

    else:
        pretrained_model = BaselineMLP(args).to(device)
        pretrained_model.load_state_dict(torch.load(os.path.join(args.model_save_pth, f"baselinemlp_pretrain_{args.seed}_{args.pretrain_remarks}.pth"),map_location=device))
        logger.info(f"pretrained_model: baselinemlp_pretrain_{args.seed}_{args.pretrain_remarks}.pth")
        pretrained_classifier = None
    # trainer
    stream_trainer = StreamTrainer(args, pretrained_model, pretrained_classifier, device, train_stream_dataloader, test_stream_dataloader)
    res = stream_trainer.train()
    return res