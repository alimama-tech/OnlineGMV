import os
import torch
from logging import getLogger
from method_src.data import get_stream_dataset, get_pretrain_bdl_stream_dataset, get_stream_with_prev_dataset
from method_src.dataset import StreamDataLoader, Pretrain_BDL_StreamDataLoader, With_GA_StreamDataLoader
from method_src.models.Classifier import BaseMLP
from method_src.models.Calibrator import CalibratorWithTime
from method_src.models.TwoTower import SharedBottomTwoTower
from method_src.stream_trainer_moe import StreamTrainerMoe
from method_src.stream_trainer_bdl import StreamTrainerBDL
from method_src.models.BaselineMLP import BaselineMLP

    

logger = getLogger()

def run_stream(args,device):
    # data
    if "bdl" in args.model:
        train_stream, test_stream = get_pretrain_bdl_stream_dataset(args)
        train_stream_dataloader = Pretrain_BDL_StreamDataLoader(train_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_stream_dataloader = Pretrain_BDL_StreamDataLoader(test_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif "reader" in args.model:
        logger.info("Using READER stream")
        train_stream, test_stream = get_stream_with_prev_dataset(args)
        train_stream_dataloader = With_GA_StreamDataLoader(train_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_stream_dataloader = With_GA_StreamDataLoader(test_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        train_stream, test_stream = get_stream_dataset(args)
        train_stream_dataloader = StreamDataLoader(train_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_stream_dataloader = StreamDataLoader(test_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if "baselinemlp" in args.model:
        pretrained_model = BaselineMLP(args).to(device)
        pretrained_model.load_state_dict(torch.load(os.path.join(args.model_save_pth, f"baselinemlp_pretrain_{args.seed}_all_pay.pth"),map_location=device))
        logger.info(f"pretrained_model: baselinemlp_pretrain_{args.seed}_all_pay.pth")
        logger.info(f"Pretrained Model:\n {pretrained_model}")
        pretrained_classifier = None
        pretrained_calibrator = None
    else:
        pretrained_model = SharedBottomTwoTower(args).to(device)
        pretrained_classifier = BaseMLP(args).to(device)
        pretrained_calibrator = CalibratorWithTime(args).to(device)
        pretrained_calibrator.load_state_dict(torch.load(os.path.join(args.model_save_pth, f"calibrator_pretrain_{args.seed}_calibrator.pth"),map_location=device))
        logger.info(f"Pretrained Calibrator:calibrator_pretrain_{args.seed}_calibrator.pth")

        pretrained_model.load_state_dict(torch.load(os.path.join(args.model_save_pth, f"sharebottom_twotower_pretrain_{args.seed}_sharedtwotower.pth"),map_location=device))
        pretrained_classifier.load_state_dict(torch.load(os.path.join(args.model_save_pth, f"classifier_pretrain_{args.seed}_classifier.pth"),map_location=device))
        logger.info(f"pretrained_model: sharebottom_twotower_pretrain_{args.seed}_sharedtwotower.pth")
        logger.info(f"Pretrained Classifier:classifier_pretrain_{args.seed}_classifier.pth")
        
    if "bdl" in args.model:
        stream_trainer = StreamTrainerBDL(args, pretrained_model, pretrained_classifier, pretrained_calibrator, device, train_stream_dataloader, test_stream_dataloader)
    else:
        # trainer
        stream_trainer = StreamTrainerMoe(args, pretrained_model, pretrained_classifier, pretrained_calibrator, device, train_stream_dataloader, test_stream_dataloader)
    
    res = stream_trainer.train()
    return res