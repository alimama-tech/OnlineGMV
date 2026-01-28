from method_src.data import get_pretrain_dataset, get_pretrain_calibrator_dataset
from method_src.dataset import TrainingDataset, EvaluationDataset, CalibratorEvaluationDataset, CalibratorTrainingDataset
from torch.utils.data import DataLoader
from logging import getLogger
from method_src.trainer import PretrainedTrainer
from method_src.models.Classifier import BaseMLP
from method_src.models.Calibrator import CalibratorWithTime


logger = getLogger()

def run(args, device):
    if "calibrator" in args.model:
        logger.info("Using calibrator data")
        dataset = get_pretrain_calibrator_dataset(args)
    else:
        dataset = get_pretrain_dataset(args)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    if "calibrator" in args.model:
        train_dataset = CalibratorTrainingDataset(train_dataset)
        test_dataset = CalibratorEvaluationDataset(test_dataset)
    else:
        train_dataset = TrainingDataset(train_dataset)
        test_dataset = EvaluationDataset(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
 
    if "classifier" in args.model:
        model = BaseMLP(args).to(device)
    elif "calibrator" in args.model:
        model = CalibratorWithTime(args).to(device)
    else:
        raise ValueError(f"Unsupported pretrain model: {args.model}")
    
    logger.info(f"Model:\n {model}")
    # get trainer
    trainer = PretrainedTrainer(args, model, device)
    # train
    logger.info("Start training...")
    trainer.train(train_dataloader, test_dataloader)
