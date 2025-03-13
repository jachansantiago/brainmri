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
from binary import hd
import argparse
import sys
import time

random.seed(42)

def get_max_memory_allocated():
    return torch.cuda.max_memory_allocated(0) / 1024 ** 2

def val_epoch(model, val_loader, criterion, metric, dice_metric_batch, post_trans, device):
    model.eval()

    val_loss = 0
    metric.reset()
    dice_metric_batch.reset()
    sample_count = 0
    hausdorff = 0
    hausdorff_tc = 0
    hausdorff_wt = 0
    hausdorff_et = 0
    avg_time = 0


    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", total=len(val_loader)):
            tic = time.time()
            x = batch["image"]
            y = batch["label"]
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            val_loss += loss.item()

            post_y_pred = post_trans(y_pred)
            metric(post_y_pred, y)
            dice_metric_batch(post_y_pred, y)
            hausdorff += hd(post_y_pred, y)
            hausdorff_tc += hd(post_y_pred[:, 0:1], y[:, 0:1])
            hausdorff_wt += hd(post_y_pred[:, 1:2], y[:, 1:2])
            hausdorff_et += hd(post_y_pred[:, 2:], y[:, 2:])
            sample_count += x.shape[0]
            toc = time.time()
            avg_time += toc - tic
    dice_score = metric.aggregate().item()
    dice_metric_batch_score = dice_metric_batch.aggregate()
    metric_tc = dice_metric_batch_score[0].item()
    metric_wt = dice_metric_batch_score[1].item()
    metric_et = dice_metric_batch_score[2].item()
    hausdorff = hausdorff / sample_count
    hausdorff_tc = hausdorff_tc / sample_count
    hausdorff_wt = hausdorff_wt / sample_count
    hausdorff_et = hausdorff_et / sample_count
    avg_time = avg_time / len(val_loader)

    metrics = {
        "dice_score": dice_score,
        "dice_tc": metric_tc,
        "dice_wt": metric_wt,
        "dice_et": metric_et,
        "hausdorff": hausdorff,
        "hausdorff_tc": hausdorff_tc,
        "hausdorff_wt": hausdorff_wt,
        "hausdorff_et": hausdorff_et,
        "avg_time": avg_time,
        "loss": val_loss / len(val_loader)
    }

    return metrics


def main(args):
    set_determinism(seed=42)
    BATCH_SIZE = args.batch_size

    date = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    experiment_name = f'{date}-eval-k-fold-unet'
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
    logging.info(f'Batch size: {BATCH_SIZE}')

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

    criterion = DiceLoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

    metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


    df_dict = {
        "val_loss": [],
        "val_dice": [],
        "val_haus": [],
        "fold": []
    }

    for _k, val_dataset in enumerate(val_datasets):
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        logging.info(f"Fold {_k + 1}/{k}")
        logging.info(f"Val dataset size: {len(val_dataset)}")
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
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        net = net.to(device)
        metrics = val_epoch(net, val_loader, criterion, metric, dice_metric_batch, post_trans, device)
        val_loss = metrics["loss"]
        val_dice = metrics["dice_score"]
        val_haus = metrics["hausdorff"]
        avg_time = metrics["avg_time"]

        dice_tc = metrics["dice_tc"]
        dice_wt = metrics["dice_wt"]
        dice_et = metrics["dice_et"]

        hausdorff_tc = metrics["hausdorff_tc"]
        hausdorff_wt = metrics["hausdorff_wt"]
        hausdorff_et = metrics["hausdorff_et"]
        # print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_acc:.4f}")
        df_dict["val_loss"].append(val_loss)
        df_dict["val_dice"].append(val_dice)
        df_dict["val_haus"].append(val_haus)
        df_dict["val_dice_tc"].append(dice_tc)
        df_dict["val_dice_wt"].append(dice_wt)
        df_dict["val_dice_et"].append(dice_et)
        df_dict["val_haus_tc"].append(hausdorff_tc)
        df_dict["val_haus_wt"].append(hausdorff_wt)
        df_dict["val_haus_et"].append(hausdorff_et)
        df_dict["fold"].append(k)
        gpu_memory = get_max_memory_allocated()
        logging.info(f'Fold {_k + 1}: Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val Haus: {val_haus:.4f}, Batch Time: {avg_time:.4f}, GPU Memory: {gpu_memory:.2f} MB')
        logging.info(f'Dice TC: {dice_tc:.4f}, Dice WT: {dice_wt:.4f}, Dice ET: {dice_et:.4f}')
        logging.info(f'Hausdorff TC: {hausdorff_tc:.4f}, Hausdorff WT: {hausdorff_wt:.4f}, Hausdorff ET: {hausdorff_et:.4f}')
        logging.info(f"Fold {_k + 1}/{k} completed.")
        pd.DataFrame(df_dict).to_csv(f"{experiment_folder}/results.csv", index=False)

    # compute mean and std of the results
    mean_val_loss = sum(df_dict["val_loss"]) / k
    mean_val_dice = sum(df_dict["val_dice"]) / k
    mean_val_haus = sum(df_dict["val_haus"]) / k
    std_val_loss = (sum((x - mean_val_loss) ** 2 for x in df_dict["val_loss"]) / k) ** 0.5
    std_val_dice = (sum((x - mean_val_dice) ** 2 for x in df_dict["val_dice"]) / k) ** 0.5
    std_val_haus = (sum((x - mean_val_haus) ** 2 for x in df_dict["val_haus"]) / k) ** 0.5
    logging.info(f'Mean Val Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}')
    logging.info(f'Mean Val Dice: {mean_val_dice:.4f} ± {std_val_dice:.4f}')
    logging.info(f'Mean Val Haus: {mean_val_haus:.4f} ± {std_val_haus:.4f}')

    mean_val_dice_tc = sum(df_dict["val_dice_tc"]) / k
    mean_val_dice_wt = sum(df_dict["val_dice_wt"]) / k
    mean_val_dice_et = sum(df_dict["val_dice_et"]) / k
    std_val_dice_tc = (sum((x - mean_val_dice_tc) ** 2 for x in df_dict["val_dice_tc"]) / k) ** 0.5
    std_val_dice_wt = (sum((x - mean_val_dice_wt) ** 2 for x in df_dict["val_dice_wt"]) / k) ** 0.5
    std_val_dice_et = (sum((x - mean_val_dice_et) ** 2 for x in df_dict["val_dice_et"]) / k) ** 0.5
    logging.info(f'Mean Dice TC: {mean_val_dice_tc:.4f} ± {std_val_dice_tc:.4f}')
    logging.info(f'Mean Dice WT: {mean_val_dice_wt:.4f} ± {std_val_dice_wt:.4f}')
    logging.info(f'Mean Dice ET: {mean_val_dice_et:.4f} ± {std_val_dice_et:.4f}')

    mean_val_haus_tc = sum(df_dict["val_haus_tc"]) / k
    mean_val_haus_wt = sum(df_dict["val_haus_wt"]) / k
    mean_val_haus_et = sum(df_dict["val_haus_et"]) / k

    std_val_haus_tc = (sum((x - mean_val_haus_tc) ** 2 for x in df_dict["val_haus_tc"]) / k) ** 0.5
    std_val_haus_wt = (sum((x - mean_val_haus_wt) ** 2 for x in df_dict["val_haus_wt"]) / k) ** 0.5
    std_val_haus_et = (sum((x - mean_val_haus_et) ** 2 for x in df_dict["val_haus_et"]) / k) ** 0.5
    logging.info(f'Mean Hausdorff TC: {mean_val_haus_tc:.4f} ± {std_val_haus_tc:.4f}')
    logging.info(f'Mean Hausdorff WT: {mean_val_haus_wt:.4f} ± {std_val_haus_wt:.4f}')
    logging.info(f'Mean Hausdorff ET: {mean_val_haus_et:.4f} ± {std_val_haus_et:.4f}')

    print("Done.")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    main(args)
