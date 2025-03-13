import os
import random
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from datasets import get_dataset
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose)
import pandas as pd
from monai.utils import set_determinism
# import onnxruntime
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader
import datetime

import argparse
import sys

random.seed(42)

def get_max_memory_allocated():
    return torch.cuda.max_memory_allocated(0) / 1024 ** 2

def train_epoch(model, train_loader, criterion, optimizer, metric, post_trans, device):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    metric.reset()
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    for batch in tqdm(train_loader, desc="Training", total=len(train_loader)):
        x = batch["image"]
        y = batch["label"]
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda',enabled=True):
            y_pred = model(x)
            loss = criterion(y_pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        post_y_pred = post_trans(y_pred)
        metric(post_y_pred, y)
    dice_score = metric.aggregate().item()

    return train_loss / len(train_loader), dice_score

def val_epoch(model, val_loader, criterion, metric, post_trans, device):
    model.eval()

    val_loss = 0
    correct = 0
    total = 0
    metric.reset()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", total=len(val_loader)):
            x = batch["image"]
            y = batch["label"]
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            val_loss += loss.item()

            post_y_pred = post_trans(y_pred)
            metric(post_y_pred, y)
    dice_score = metric.aggregate().item()
    return val_loss / len(val_loader), dice_score


def main(args):
    set_determinism(seed=42)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    date = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    experiment_name = f'{date}-k-fold-unet'
    experiment_folder = f'exps/{experiment_name}'
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    log_file = f'{experiment_folder}/log.txt'
    # set logging with INFO level and date format 
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] - %(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ])

    logging.info(f'Experiment: {experiment_name}')
    logging.info(f'Epochs: {EPOCHS}')
    logging.info(f'Batch size: {BATCH_SIZE}')
    logging.info(f'Learning rate: {LR}')

    root_dir = '.'
    train_dataset = get_dataset(root_dir, train=True)

    # K-fold cross-validation
    k = 5
    fold_size = len(train_dataset) // k
    train_indices = list(range(len(train_dataset)))
    train_datasets = []
    val_datasets = []
    # Shuffle the indices
    random.shuffle(train_indices)

    # Split the dataset into k folds
    for fold in range(k):
        # Create train and validation sets
        val_indices = train_indices[fold * fold_size: (fold + 1) * fold_size]
        train_indices_fold = train_indices[:fold * fold_size] + train_indices[(fold + 1) * fold_size:]

        # save the train and val indices
        train_indices_fold_path = f"{experiment_folder}/train_indices_fold_{fold}.txt"
        val_indices_path = f"{experiment_folder}/val_indices_{fold}.txt"
        train_idx_df = pd.DataFrame(train_indices_fold)
        val_idx_df = pd.DataFrame(val_indices)
        train_idx_df.to_csv(train_indices_fold_path, index=False, header=False)
        val_idx_df.to_csv(val_indices_path, index=False, header=False)


        dataset_train = torch.utils.data.Subset(train_dataset, train_indices_fold)
        dataset_val = torch.utils.data.Subset(train_dataset, val_indices)
        train_datasets.append(dataset_train)
        val_datasets.append(dataset_val)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = BATCH_SIZE
    epochs = EPOCHS

    criterion = DiceLoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

    metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


    df_dict = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
        "epoch": [],
        "fold": []
    }

    for _k, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        logging.info(f"Fold {_k + 1}/{k}")
        logging.info(f"Train dataset size: {len(train_dataset)}")
        net = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,  
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2), 
            num_res_units=3 
        )
        if args.resume:
            resume_exp_folder = args.resume
            model_path = f"{resume_exp_folder}/model_fold_{_k}.pth"
            if os.path.exists(model_path):
                net.load_state_dict(torch.load(model_path, weights_only=True))
                logging.info(f"Loaded model from {model_path}")
            else:
                logging.info(f"Model not found at {model_path}, starting from scratch.")
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        net = net.to(device)
        best_dice = 0
        
        for epoch in range(epochs):
            # logging.info(f"Epoch {epoch + 1}/{epochs}")
            train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, metric, post_trans, device)
            # print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_acc:.4f}")

            val_loss, val_acc = val_epoch(net, val_loader, criterion, metric, post_trans, device)
            # print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_acc:.4f}")
            df_dict["train_loss"].append(train_loss)
            df_dict["train_dice"].append(train_acc)
            df_dict["val_loss"].append(val_loss)
            df_dict["val_dice"].append(val_acc)
            df_dict["epoch"].append(epoch)
            df_dict["fold"].append(k)
            gpu_memory = get_max_memory_allocated()
            logging.info(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Dice: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_acc:.4f}, GPU Memory: {gpu_memory:.2f} MB')
            if val_acc > best_dice:
                best_dice = val_acc
                torch.save(net.state_dict(), f"{experiment_folder}/best_model_{_k}.pth")

            # scheduler.step()
        logging.info(f"Fold {_k + 1}/{k} completed.")
        logging.info("Saving model...")
        torch.save(net.state_dict(), f"{experiment_folder}/model_fold_{_k}.pth")
        logging.info("Model saved.")
        pd.DataFrame(df_dict).to_csv(f"{experiment_folder}/results.csv", index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    main(args)
