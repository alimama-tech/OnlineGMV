from src.data import get_pretrain_dataset
from src.dataset import TrainingDataset, EvaluationDataset
from torch.utils.data import DataLoader
from src.models.BaselineMLP import BaselineMLP
from logging import getLogger
from src.trainer import PretrainedTrainer
from src.models.TwoTower import SharedBottomTwoTower

logger = getLogger()

def run(args, device):
    """
    Run the pretraining process.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        device (torch.device): Device to run the model on.
    """
    dataset = get_pretrain_dataset(args)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset = TrainingDataset(train_dataset)
    test_dataset = EvaluationDataset(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # get model
    if "twotower" in args.model:
        model = SharedBottomTwoTower(args).to(device)
    else:
        model = BaselineMLP(args).to(device)
    logger.info(f"Model:\n {model}")
    # get trainer
    trainer = PretrainedTrainer(args, model, device)
    # train
    logger.info("Start training...")
    trainer.train(train_dataloader, test_dataloader)
