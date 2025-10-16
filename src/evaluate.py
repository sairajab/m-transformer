import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
from models import SampleLevelRegressor, BasicRegressorWithASVEncoder
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
#from dataset import load_data, sample_test_data
import os 
from glob import glob
import re 
import seaborn as sns
import matplotlib.pyplot as plt
from model_orig import BasicRegressor, BasicRegressorwithUnifrac
import json
import shutil
from dataset_sparse import collate_fn as sparse_collate_fn
from torch.utils.data import Dataset, DataLoader
from dataset import DataProcessor, Arguments
def _mean_absolute_error(pred_val, true_val, fname, labels=None):
    pred_val = np.squeeze(pred_val)
    true_val = np.squeeze(true_val)
    mae = np.mean(np.abs(true_val - pred_val))

    min_x = np.min(true_val)
    max_x = np.max(true_val)
    coeff = np.polyfit(true_val, pred_val, deg=1)
    p = np.poly1d(coeff)
    xx = np.linspace(min_x, max_x, 50)
    yy = p(xx)

    diag = np.polyfit(true_val, true_val, deg=1)
    p = np.poly1d(diag)
    diag_xx = np.linspace(min_x, max_x, 50)
    diag_yy = p(diag_xx)
    data = {"pred": pred_val, "true": true_val}
    data = pd.DataFrame(data=data)
    plot = sns.scatterplot(data, x="true", y="pred")
    plt.plot(xx, yy)
    plt.plot(diag_xx, diag_yy)
    mae = "%.4g" % mae
    plot.set(xlabel="True")
    plot.set(ylabel="Predicted")
    plot.set(title=f"MAE: {mae}")
    plt.savefig(fname)
    plt.close()
    return mae


def predict(model, loader, device, multitask=False):
    """Generate predictions for a dataset"""
    model = model.to(device)
    model.eval()
    predictions = []
    labels = []
    criterion = nn.MSELoss()
    test_loss = 0
    all_ids = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embeddings'].to(device, dtype=torch.float32)
            abundances = batch['abundances'].to(device)
            masks = batch['masks'].to(device)
            targets = batch['outdoor_add_0'].to(device)
            all_ids.extend(batch["SampleID"])
            env = batch['env'].to(device)  # 0 for indoor, 1 for outdoor
            # print("Num non-zero ASVs per sample:", (abundances > 0).sum(-1).float().mean())
            # print("Top-1 abundance value:", abundances.max(-1).values.mean())


            outputs, _ , _ = model(embeddings, abundances, masks)
            if multitask:
                p_indoor = outputs[0]
                p_outdoor = outputs[1]
                mask_in = (env == 0).view(-1)
                mask_out = (env == 1).view(-1)
                all_outputs = torch.cat([p_indoor[mask_in], p_outdoor[mask_out]])
                all_targets = torch.cat([targets[mask_in], targets[mask_out]])
                outputs = all_outputs
                targets = all_targets
                
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()        
            predictions.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
            
    # test_loss /= len(loader)
    
    # print("Loss ", test_loss)

    print("Unique SampleIDs in predict():", len(set(all_ids)))
    print("Expected SampleIDs:", len(loader)) 
    return np.concatenate(labels), np.concatenate(predictions)
def get_residual_plot(predictions, labels, fname):
    # Residuals
    residuals = (np.array(predictions) - np.array(labels)) / 100.0  # Scale back to original values
    y_pred = np.array(predictions) / 100.0  # Scale back to original values
    # Option 1: residuals vs prediction
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals (y_pred - y_true)")
    plt.title("Residual Plot")
    plt.savefig(f"{res}/residuals_vs_prediction.png")
    plt.show()
        
def evaluate_asv_encoder(donor_ids, res="results/", find_best = True):
    train_asv_encoder = True
    one_hot_seqs = None
    embedding_path = None
    eval_runs = 3
    all_labels = []
    all_predictions = []
    results = defaultdict(float)
    mean_mae = 0
    for donor_id in donor_ids: 
        for _ in range(eval_runs):
            abundance_table, train_data , test_data, one_hot_seqs, distances = load_data(heldout = donor_id, train_encoder=True)
            test_loader = sample_test_data(abundance_table, test_data, embedding_path = embedding_path, one_hot_seqs=one_hot_seqs, random_vector=random_vec) 
            runs = 2   
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            best_mae = 1000
            best_i = 0
            for i in range(1, runs+1):
                dir = f"{res}/{donor_id}/run_{i}/"
                with open(dir + 'res.json') as f:
                    config = json.load(f)
                    mae = float(config['best_mae'])
                    print(f"MAE for {donor_id} : {mae}")
                if mae < best_mae:
                    best_mae = mae
                    best_run = dir
                    best_i = i                

            model_files = best_run + "model.pt"
            print(f"Best run: {best_run}", best_mae)
            print(f"Done for {donor_id} : {best_mae}")
            model = SampleLevelRegressor(use_nt_encoder=train_asv_encoder, pe=True)

            ## print model parameters
            print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            model.load_state_dict(torch.load(model_files))
            #print(model)
            labels, predictions = predict(model, test_loader , device)
                
            all_labels.extend(labels*100)
            all_predictions.extend(predictions*100)
                
            mae = _mean_absolute_error(predictions*100, labels*100,f'{best_run}/test.png')
            print(f'Test MAE run {best_i}: {mae} : {model._count_alpha}')
            mae_f = float(mae)
            results[donor_id] += mae_f
                
        results[donor_id] = results[donor_id]/eval_runs
        mean_mae += results[donor_id]
        _mean_absolute_error( all_predictions, all_labels,f'{res}/test_all.png')
        pd.DataFrame.from_dict(results, orient='index').to_csv(f"{res}/orig_results.csv", index=True)
        
        get_residual_plot(all_predictions, all_labels, f"{res}/residuals.png")
        print(f"Mean MAE: {mean_mae / len(donor_ids)}")
        results["mean_mae"] = mean_mae / len(donor_ids)
        # save in file
        with open(f"{res}/results.json", 'w') as f:
            json.dump(results, f, indent=4)
            
def evaluate_all_runs(donor_ids, res,  metadata_file, biom_file, embedding_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for donor_id in donor_ids:
        print(f"\n=== Donor {donor_id} ===")

        args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=donor_id,
                )
        data_processor = DataProcessor(args)
        data_processor.load_data(multitask=True)
            
        test_dataset = data_processor.sample_test_data()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=sparse_collate_fn)
            

        # evaluate every run
        for i in range(1, 4):  # adjust number of runs
            run_dir = f"{res}/{donor_id}/run_{i}/"
            model_file = run_dir + "model.pt"

            model = BasicRegressor(input_dim=128, pe=True).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()

            labels, predictions = predict(model, test_loader, device, multitask=True)

            mae_t = float(_mean_absolute_error(predictions*100, labels*100, f'{run_dir}/test.png'))
            print(f"Donor {donor_id} | Run {i} | Test MAE: {mae_t:.4f}")

            # (optional) also save per-run results
            with open(run_dir + "test_eval.json", "w") as f:
                json.dump({"test_mae": float(mae_t)}, f, indent=2)


def evaluate_basic(donor_ids, res, metadata_file, biom_file, embedding_path):           
    results = {}
    all_labels = []
    all_predictions = []
    eval_runs = 7
    runs = 3

    mean_mae = 0
    
    for donor_id in donor_ids:
        donor_labels = []
        donor_preds = []

        # ---- Find best run (lowest val MAE) ----
        best_mae = float("inf")
        best_run_dir = None

        for i in range(1, runs+1):
            dir = f"{res}/{donor_id}/run_{i}/"
            with open(dir + 'res.json') as f:
                config = json.load(f)
                mae = float(config['best_mae'])
                if mae < best_mae:
                    best_mae = mae
                    best_run_dir = dir

        print(f"Best run for donor {donor_id}: {best_run_dir} (val MAE {best_mae})")

        # ---- Evaluate best run multiple times ----
        total_mae = 0
        for _ in range(eval_runs):
            args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=donor_id,
                )
            data_processor = DataProcessor(args)
            data_processor.load_data(multitask=True)
            
            test_dataset = data_processor.sample_test_data()
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=sparse_collate_fn)
            

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = BasicRegressor(input_dim=128, pe=True)
            model.load_state_dict(torch.load(best_run_dir + "model.pt"))
            model.to(device)

            labels, predictions = predict(model, test_loader, device, multitask=True)

            mae_t = _mean_absolute_error(predictions*100, labels*100, f'{best_run_dir}/test.png')
            total_mae += float(mae_t)

            donor_labels.extend(labels*100)
            donor_preds.extend(predictions*100)

        avg_mae = total_mae / eval_runs
        results[donor_id] = avg_mae
        mean_mae += avg_mae

        # collect global predictions for residual plot
        all_labels.extend(donor_labels)
        all_predictions.extend(donor_preds)

    # ---- After all donors ----
    _mean_absolute_error(all_predictions, all_labels, f'{res}/test_all.png')
    get_residual_plot(all_predictions, all_labels, f"{res}/residuals.png")

    results["mean_mae"] = mean_mae / len(donor_ids)
    pd.DataFrame.from_dict(results, orient='index').to_csv(f"{res}/orig_results.csv", index=True)

    with open(f"{res}/results.json", 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Mean MAE: {results['mean_mae']}")

 

if __name__ == "__main__":

        from sklearn.model_selection import train_test_split
        folds = 3
        random_vec = False
#         donor_ids = ['D19','D7','D8','D22','D13','D15','D28','D10','D17','D11','D4','D26','D23','D29','D27','D20','D6',
#  'D25','D30','D5','D21','D18','D14','D12','D24','D9','D16']
        #donor_ids = ["D27", "D15","D12", "D24"]#, "D15",'D19', "D22", "D27", ]  #
        donor_ids =  ['D4', 'D17', 'D29', 'D12', 'D6','D8', 'D10', 'D13', 'D15', 'D19', 'D21', 'D23', 'D25', 'D27','D11']


        res = "finetune_dnabert_results/all_outdoors/"
        biom_file = "../data/new_data/table_all/feature-table.biom"
        metadata_file = "../data/new_data/metadata_all_outdoor.tsv"
        embedding_path = "../data/embeddings/al_outdoors.h5"
        
        res = "finetune_dnabert_results/multitask/"
        biom_file = "../data/new_data/feature-table/feature-table.biom"
        metadata_file = "../data/new_data/metadata_sheds.tsv"
        embedding_path = "../data/embeddings/sheds.h5"
        donor_ids = pd.read_csv(metadata_file, sep="\t")['DonorID'].unique().tolist()
        evaluate_basic(donor_ids, res, metadata_file, biom_file, embedding_path)
        evaluate_all_runs(donor_ids, res, metadata_file, biom_file, embedding_path)
