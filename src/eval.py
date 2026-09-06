import os
import time
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from contextlib import suppress
from tqdm import tqdm

# Project-specific imports
from dataloader import *
from modules import (
    attmil, clam, dsmil, transmil, mean_max,
    attmil_ibmil
)
from utils import *

# Optional tracking
# import wandb
# wandb.login(key=[your_api_key])
def build_model(args, device, k=None):
    """
    Helper to build and return the model specified in args.model
    """
    model_name = args.model.lower()
    
    if model_name == 'attmil':
        model = attmil.DAttention(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout, act=args.act)
  
    elif model_name == 'gattmil':
        model = attmil.AttentionGated(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout)
    
    elif model_name == 'cf_gattmil_cal': # at inference copy only factual model and discard cf modules
        model = attmil.AttentionGated(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout)
        #attmil_cf.AttentionGated_CAL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout).to(device)

    elif args.model == 'cf_gattmil_cal_u':
        model = attmil.AttentionGated(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout)
            
    elif model_name == 'inst_gattmil':
        model = attmil.Inst_AttentionGated(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout)
    
    elif model_name == 'add_gattmil':
        model = attmil.Add_AttentionGated(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout)

    elif model_name == 'ibmil':
        if not args.confounder_path.endswith('.npy'):
            confounder_path = os.path.join(
                args.confounder_path, str(k),
                f"train_bag_cls_agnostic_feats_proto_{args.confounder_k}.npy")
        else:
            confounder_path = args.confounder_path

        model = attmil_ibmil.Dattention_ori(
            out_dim=args.n_classes, dropout=args.dropout,
            in_size=args.input_dim, confounder_path=confounder_path)

    elif model_name == 'clam_sb':
        model = clam.CLAM_SB(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout, act=args.act)

    elif model_name == 'clam_mb':
        model = clam.CLAM_MB(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout, act=args.act)

    elif model_name == 'transmil':
        model = transmil.TransMIL(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout, act=args.act)

    elif model_name == 'dsmil':
        model = dsmil.MILNet(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout, act=args.act)
        args.cls_alpha = 0.5
        args.aux_alpha = 0.5

    elif model_name == 'meanmil':
        model = mean_max.MeanMIL(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout, act=args.act)

    elif model_name == 'maxmil':
        model = mean_max.MaxMIL(
            input_dim=args.input_dim, n_classes=args.n_classes,
            dropout=args.dropout, act=args.act)

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Move model to device
    model = model.to(device)
    return model

def main(args):
    # set seed
    seed_torch(args.seed)

    # --->generate dataset
    if args.datasets.lower() == 'camelyon16':
        label_path=os.path.join(args.dataset_root,'metadata.csv')
        p, l = get_patient_label_c16(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    elif args.datasets.lower() == 'tcga':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    elif args.datasets.lower() == 'luad_tp53':
        label_path=os.path.join(args.dataset_root,'metadata.csv')
        p, l = get_patient_label_luad(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    if args.cv_fold > 1:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)

    acs, pre, rec,fs,auc=[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc]

    print('Dataset: ' + args.datasets)

    for k in range(args.fold_start, args.cv_fold)[args.task_id:args.task_id+1]: # from 0 to 4
        print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        res_one_fold = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)
        ckc_metric = res_one_fold['ckc_metric']
        res_flapping = res_one_fold['flapping']

        res_flapping.to_csv(os.path.join(dir_eval_flap, f'flapping_results_{args.approach}_{args.attribution_strategy}_{args.evidence}_fold_{k}.csv'))
        
        if args.approach == 'drop':
            metrics_df = pd.DataFrame({
                'fold': [k],
                'model':[args.model],
                'project':[args.project],
                'accuracy': acs,
                'auc': auc,
                'precision': pre,
                'recall': rec,
                'fscore': fs
                })

            metrics_df.to_csv(os.path.join(dir_eval_flap, 'cv_metrics_fold_{fold}.csv').format(fold=k))


 
    if True: #not args.no_log:
        print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

def one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l):
    # ---> Initialization
    seed_torch(args.seed)
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    acs,pre,rec,fs,auc = ckc_metric

    # ---> Loading data
    if args.datasets.lower() == 'camelyon16':

        train_set = C16Dataset(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
        test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        if args.val_ratio != 0.:
            val_set = C16Dataset(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        else:
            val_set = test_set

    elif args.datasets.lower() == 'tcga':
        
        train_set = TCGADataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True,_type=args.tcga_sub)
        test_set = TCGADataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
        if args.val_ratio != 0.:
            val_set = TCGADataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
        else:
            val_set = test_set
    
    elif args.datasets.lower() == 'luad_tp53':
        
        train_set = TCGADataset_LUAD(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True,_type=args.tcga_sub)
        test_set = TCGADataset_LUAD(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
        if args.val_ratio != 0.:
            val_set = TCGADataset_LUAD(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
        else:
            val_set = test_set

    if args.fix_loader_random:
        # generated by int(torch.empty((), dtype=torch.int64).random_().item())
        big_seed_list = 7784414403328510413
        generator = torch.Generator()
        generator.manual_seed(big_seed_list)  

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # bulid networks

    model = build_model(args, device, k=None)

    if args.fix_train_random:
        seed_torch(args.seed)


    # test
    best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
    info = model.load_state_dict(best_std['model'], strict=False) # to adapt to post training/adaptation with CF setting
    print('Loading best ckp ', info)
    results_test = test(args,model,test_loader,device,k)
    accuracy, auc_value, precision, recall, fscore = results_test['Classification Eval']
    wsi_ids, predicted_probs_all,entropy_all, gini_all, false_pred_all = results_test['Patch Flapping']

    res_flapping = pd.DataFrame(data={
                    'slide_id': wsi_ids,
                    'predicted_probs':predicted_probs_all,
                    'entropy-curve': entropy_all,
                    'gini_curve': gini_all,
                    'false_pred': false_pred_all,
                })
    
 
    print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (accuracy, auc_value, precision, recall, fscore))
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)

    return {'ckc_metric': [acs,pre,rec,fs,auc] , 'flapping':res_flapping}

def test(args, model, loader, device,k):
    
    model.eval()
    wsi_ids, bag_logits, bag_labels = [], [], []
    predicted_probs_all, false_pred_all = [], []
    entropy_all, gini_all = [], []

    approach = args.approach  # 'drop' or 'add'
    strategy = args.strategy  # e.g., '1%-of-all', 'one-by-one'

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            wsi_ids.append(i)
            label = data[1].to(device)
            if len(label) > 1:
                bag_labels.extend(label.tolist())
            else:
                bag_labels.append(label.item())

            if isinstance(data[0], (list, tuple)):
                for j in range(len(data[0])):
                    data[0][j] = data[0][j].to(device)
                bag = data[0]
                batch_size = data[0][0].size(0)
            else:
                bag = data[0].to(device)
                batch_size = bag.size(0)

            # Forward pass 
                
            if args.model == 'dsmil':
                test_logits, _, attention_heatmap = model.forward_eval(bag)
            elif 'cf' in args.model:
                test_logits_all, attention_heatmap_all = model.forward_eval(bag)
                test_logits = test_logits_all[0].view(batch_size,-1)
                if args.evidence == 'regular':
                    attention_heatmap =  attention_heatmap_all[0]
                elif args.evidence == "mean":
                    attention_heatmap =  attention_heatmap_all.mean(axis=0)
                elif args.evidence == "max":
                    attention_heatmap, _ =  attention_heatmap_all.max(axis=0)
                elif args.evidence == "delta":
                    attention_heatmap =  attention_heatmap_all[0] - attention_heatmap_all[1]
                
            else:
                test_logits, attention_heatmap = model.forward_eval(bag)
            
            if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                probs_orig = (0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1)).cpu().numpy()
                
            else:
        
                probs_orig = torch.softmax(test_logits,dim=-1).cpu().numpy()
            
            bag_logits.append(probs_orig[0][1]) #get logit for class 1
            #### Patch flapping eval
            # Attribution strategy
            if args.attribution_strategy == 'random':
                # patch_scores = np.random.randn(attention_heatmap.shape[-1])
                patch_scores = torch.rand_like(attention_heatmap).squeeze().cpu().numpy()
            else:
                patch_scores = attention_heatmap.squeeze().cpu().numpy()
                if args.attribution_strategy == 'absolute':
                    patch_scores = np.abs(patch_scores)
            
            # --- Normalize attention (softmax) ---
            attn = np.exp(patch_scores - np.max(patch_scores))
            attn /= attn.sum() + 1e-8

            # --- Compute entropy and Gini ---
            attn_entropy_base = -np.sum(attn * np.log(attn + 1e-12))
            attn_gini_base = 1 - 2 * np.trapz(np.cumsum(np.sort(attn))) / len(attn)

            # Sorting patches
            ind_sorted = np.argsort(patch_scores)
            if args.order == 'morf':
                ind_sorted = ind_sorted[::-1]

            n_patches = len(patch_scores)

            # Create a baseline (all zero) input
            if isinstance(bag, list):
                bag_zeros = deepcopy(bag)
                bag_zeros[0] = torch.zeros_like(bag[0])
            else:
                bag_zeros = torch.zeros_like(bag)
                

            if args.model == 'dsmil':
                probs_zero_logits, _, _ = model.forward_eval(bag_zeros)
            
            elif  'cf' in args.model:
                probs_zero_logits_all, _ = model.forward_eval(bag)
                probs_zero_logits = probs_zero_logits_all[0].view(batch_size,-1)
            else:
                probs_zero_logits, _ = model.forward_eval(bag_zeros)
            
            if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(probs_zero_logits,(list,tuple))):
                probs_zero = (0.5*torch.softmax(probs_zero_logits[1],dim=-1)+0.5*torch.softmax(probs_zero_logits[0],dim=-1)).squeeze().numpy()
            else:
                probs_zero = torch.softmax(probs_zero_logits,dim=-1).cpu().numpy()
            

            # Determine target class
            target_class = label.item() # here we assume label is one item ie binary classification
            false_pred = probs_orig[0, target_class].item() <= 0.5
            false_pred_all.append(false_pred)

            
            # Initialize prediction tracking
            if approach == 'drop':
                predicted_probs = [probs_orig[0, target_class].item()]
                attn_entropy_curve = []
                attn_gini_curve = []
                attn_entropy_curve.append(attn_entropy_base)
                attn_gini_curve.append(attn_gini_base)
                
            else:
                predicted_probs = [probs_zero[0, target_class].item()]
                attn_entropy_curve = []
                attn_gini_curve = []

            # Determine how many patches to drop/add in each step
            if '%-of-all' in strategy:
                perc = int(strategy[:strategy.index('%')])
                perc = np.arange(perc, 101, perc)
                if perc[-1] != 100:
                    perc = np.append(perc, 100)
                percentiles = np.percentile(patch_scores, perc)
                bins = np.append(patch_scores.min(), percentiles)
                n_drop_array, _ = np.histogram(patch_scores, bins=bins)
            elif strategy == 'one-by-one':
                n_drop_array = np.ones(n_patches, dtype=int)

            ind_add = []
            flag_empty_bag = False
            flag_full_bag = False

            for n_drop in n_drop_array:
                ind_add += list(ind_sorted[:n_drop])
                ind_sorted = np.delete(ind_sorted, np.s_[:n_drop])
                bag_ = deepcopy(bag)
                if isinstance(bag, list):
                    features = bag[0]
                else:
                    features = bag
                # Select features to keep
                

                if approach == 'drop':
                    if ind_sorted.size > 0:
                        kept_indices = sorted(ind_sorted)
                        features_mod = features[:, kept_indices, :]
                        bag_size = len(kept_indices)
                    else:
                        flag_empty_bag = True
                        features_mod = torch.zeros_like(features)
                        bag_size = n_patches
                else:  # approach == 'add'
                    if len(ind_add) == n_patches:
                        flag_full_bag = True
                        features_mod = features
                        bag_size = n_patches
                    else:
                        kept_indices = sorted(ind_add)
                        features_mod = features[:, kept_indices, :]
                        bag_size = len(kept_indices)

                if isinstance(bag, list):
                    bag_[0] = features_mod
                    input_mod = bag_
                else:
                    input_mod = features_mod

                if flag_empty_bag:
                    probs = probs_zero
                elif flag_full_bag:
                    probs = probs_orig
                    ## add base attn stats
                    attn_entropy_curve.append(attn_entropy_base)
                    attn_gini_curve.append(attn_gini_base)
                else:
                    if args.model == 'dsmil':
                        probs_logits, _, flap_attention_heatmap = model.forward_eval(input_mod)
                    elif 'cf' in args.model:
                        probs_logits_all, flap_attention_heatmap_all = model.forward_eval(input_mod)
                        probs_logits = probs_logits_all[0].view(batch_size,-1)

                        if args.evidence == 'regular':
                            flap_attention_heatmap =  flap_attention_heatmap_all[0]
                        elif args.evidence == "mean":
                            flap_attention_heatmap =  flap_attention_heatmap_all.mean(axis=0)
                        elif args.evidence == "max":
                            flap_attention_heatmap, _ =  flap_attention_heatmap_all.max(axis=0)
                        elif args.evidence == "delta":
                            flap_attention_heatmap =  flap_attention_heatmap_all[0] - flap_attention_heatmap_all[1]


                    else:
                        probs_logits, flap_attention_heatmap = model.forward_eval(input_mod)
                    
                    flap_patch_scores = flap_attention_heatmap.squeeze().cpu().numpy()
                    
                    if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(probs_logits,(list,tuple))):
                        probs = (0.5*torch.softmax(probs_logits[1],dim=-1)+0.5*torch.softmax(probs_logits[0],dim=-1)).squeeze().numpy()
                    else:
                        probs = torch.softmax(probs_logits,dim=-1).cpu().numpy()
                    
                    # --- Normalize attention (softmax) ---
                    attn = np.exp(flap_patch_scores - np.max(flap_patch_scores))
                    attn /= attn.sum() + 1e-8

                    # --- Compute entropy and Gini ---
                    attn_entropy = -np.sum(attn * np.log(attn + 1e-12))
                    attn_gini = 1 - 2 * np.trapz(np.cumsum(np.sort(attn))) / len(attn)

                    attn_entropy_curve.append(attn_entropy)
                    attn_gini_curve.append(attn_gini)
                    

                predicted_probs.append(probs[0, target_class].item())
                    
            predicted_probs_all.append(predicted_probs)
            entropy_all.append(attn_entropy_curve)
            gini_all.append(attn_gini_curve)


    # Main classification evaluation
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logits, not args.datasets.lower() == 'camelyon16' )
        
    test_results = {'Classification Eval': [accuracy, auc_value, precision, recall, fscore],
                    'Patch Flapping': [wsi_ids, predicted_probs_all, entropy_all, gini_all, false_pred_all ] }



    return test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
    parser.add_argument('--dataset_root', default='/data/xxx/TransMIL', type=str, help='Dataset root path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    parser.add_argument('--tcga_sub', default='nsclc', type=str, help='[nsclc,brca]')
    
    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--aux_alpha', default=1.0, type=float, help='Auxiliary loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--model', default='attmil', type=str, help='Model name')
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')

    # Eval
    parser.add_argument('--attribution_strategy', default='original', type=str, choices=['original', 'random', 'absolute'],
                        help='Strategy used for patch attribution')
    parser.add_argument('--approach', default='drop', type=str, choices=['drop', 'add'],
                        help='Perturbation approach: drop or add patches based on relevance')
    parser.add_argument('--strategy', default='1%-of-all', type=str,
                        help="Patch flip strategy: 'one-by-one', '%-of-all', or 'remaining-%-perc'")
    parser.add_argument('--order', default='morf', type=str, choices=['morf', 'morl'],
                        help="Order to apply patch relevance: 'morf' (most relevant first), 'morl' (last)")
        
    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # ACMIL n_token=args.n_token, n_masked_patch=args.n_masked_patch, mask_drop=args.mask_drop
    parser.add_argument( "--n_token", type=int, default=1, help="number of attention branches in (MBA).")
    parser.add_argument( "--n_masked_patch", type=int, default=0, help="top-K instances are be randomly masked in STKIM.")
    parser.add_argument( "--mask_drop", type=float, default=0.6, help="maksing ratio in the STKIM" )
    # mhim abmil
    parser.add_argument('--baseline', default='attn', type=str, help='Baselin model [attn,selfattn]')

    # Our
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')

    # CF
    parser.add_argument('--evidence', default='regular', type=str, help='attention branch to take as evidence choices = ["regular","sum","diff"]')
    
    # Transformer
    parser.add_argument('--attn', default='rmsa', type=str, help='Inner attention')
    parser.add_argument('--pool', default='attn', type=str, help='Classification poolinp. use abmil.')
    parser.add_argument('--ffn', action='store_true', help='Feed-forward network. only for ablation')
    parser.add_argument('--n_trans_layers', default=2, type=int, help='Number of layer in the transformer')
    parser.add_argument('--mlp_ratio', default=4., type=int, help='Ratio of MLP in the FFN')
    parser.add_argument('--qkv_bias', action='store_false')
     # R-MSA
    parser.add_argument('--region_attn', default='native', type=str, help='only for ablation')
    parser.add_argument('--min_region_num', default=0, type=int, help='only for ablation')
    parser.add_argument('--region_num', default=8, type=int, help='Number of the region. [8,12,16,...]')
    parser.add_argument('--trans_dim', default=64, type=int, help='only for ablation')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the R-MSA')
    parser.add_argument('--trans_drop_out', default=0.1, type=float, help='Dropout in the R-MSA')
    parser.add_argument('--drop_path', default=0., type=float, help='Droppath in the R-MSA')
   
    # PEG or PPEG. only for alation
    parser.add_argument('--pos', default='none', type=str, help='Position embedding, enable PEG or PPEG')
    parser.add_argument('--pos_pos', default=0, type=int, help='Position of pos embed [-1,0]')
    parser.add_argument('--peg_k', default=7, type=int, help='K of the PEG and PPEG')
    parser.add_argument('--peg_1d', action='store_true', help='1-D PEG and PPEG')
    # EPEG
    parser.add_argument('--epeg', action='store_false', help='enable epeg')
    parser.add_argument('--epeg_bias', action='store_false', help='enable conv bias')
    parser.add_argument('--epeg_2d', action='store_true', help='enable 2d conv. only for ablation')
    parser.add_argument('--epeg_k', default=15, type=int, help='K of the EPEG. [9,15,21,...]')
    parser.add_argument('--epeg_type', default='attn', type=str, help='only for ablation')
    # CR-MSA
    parser.add_argument('--cr_msa', action='store_false', help='enable CR-MSA')
    parser.add_argument('--crmsa_k', default=3, type=int, help='K of the CR-MSA. [1,3,5]')
    parser.add_argument('--crmsa_heads', default=8, type=int, help='head of CR-MSA. [1,8,...]')
    parser.add_argument('--crmsa_mlp', action='store_true', help='mlp phi of CR-MSA?')

    # DAttention
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # Misc
    parser.add_argument('--title', default='default', type=str, help='Title of exp')
    parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--no_log', action='store_true', default=False, help='Without log')
    parser.add_argument('--model_path', type=str, help='Output path')

    #Parral
    parser.add_argument('--task_id', type=int, default=None, help='task id to divide work inside job array')
    args = parser.parse_args()

 
    
    # if not os.path.exists(os.path.join(args.model_path,args.project)):
    os.makedirs(os.path.join(args.model_path,args.project), exist_ok = True)
    args.model_path = os.path.join(args.model_path,args.project,args.title)
    # if not os.path.exists(args.model_path):
    os.makedirs(args.model_path, exist_ok = True)
    dir_eval_flap = os.path.join(args.model_path, 'Eval_Flapping')
    os.makedirs(dir_eval_flap, exist_ok = True)

    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    if args.model == 'clam_sb':
        args.cls_alpha= .7
        args.aux_alpha = .3
    elif args.model == 'clam_mb':
        args.cls_alpha= .7
        args.aux_alpha = .3
    elif args.model == 'dsmil':
        args.cls_alpha = 0.5
        args.aux_alpha = 0.5
    elif args.model == 'pure' or args.model == 'mhim':
        args.cl_alpha=0.

    if args.datasets.lower() == 'camelyon16':
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets.lower() == 'tcga':
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets.lower() == 'luad_tp53':
        args.fix_loader_random = True
        args.fix_train_random = True


    print(args)

    localtime = time.asctime( time.localtime(time.time()) )
    print("Start time:", localtime)
    start_time = time.time()
    
    main(args=args)
    
    end_time = time.time()
    localtime_end = time.asctime(time.localtime(end_time))
    print("End time:", localtime_end)
    print("Total runtime: {:.2f} minutes ({:.2f} seconds)".format((end_time - start_time)/60, end_time - start_time))

