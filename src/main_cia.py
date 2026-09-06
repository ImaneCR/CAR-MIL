
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader, RandomSampler
import argparse, os
from modules import attmil_cf, clam, dsmil

from torch.nn.functional import one_hot

from torch.cuda.amp import GradScaler
from contextlib import suppress
import time

from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict

from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, loss_type='bce', alpha_effect=0.0):
        super(CustomLoss, self).__init__()
        self.alpha_effect = alpha_effect
        
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, preds, targets, preds_cf=None, bag_attention=None, bag_attention_cf=None):
        loss_main = self.criterion(preds, targets)
        
        loss_reg = 0.0

        if preds_cf is not None:
            preds_effect = preds - preds_cf
            loss_reg = self.criterion(preds_effect, targets)

       
        total_loss = loss_main + self.alpha_effect * loss_reg 
        return total_loss



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
        print('p' ,p[0],'l',l[0])

    if args.cv_fold > 1:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)

    acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]

    print('Dataset: ' + args.datasets)

    for k in range(args.fold_start, args.cv_fold)[args.task_id:args.task_id+1]: # from 0 to 4
        print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)

 
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
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric

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
        print('train_set',len(train_set))
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
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # bulid networks


    if args.model == 'cia_gattmil':
        model = attmil_cf.AttentionGated_CAL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout).to(device)

    elif (args.model == 'cia_clam_sb') or (args.model == 'clam_sb'):
        model = clam.CLAM_SB_CAL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    
    elif (args.model == 'cia_dsmil') or (args.model == 'dsmil') :
        model = dsmil.MILNet_CAL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
   



    if args.init_ckpt != 'none': # init model from trained ckpt

        if not args.init_ckpt.endswith('.pt'):
            _str = 'fold_{fold}_model_best_auc.pt'.format(fold=k)
            _init_ckpt = os.path.join(args.init_ckpt,_str)
        else:
            _init_ckpt =args.init_ckpt

        ckpt_trained = torch.load(_init_ckpt,map_location=device)
        model.load_state_dict(ckpt_trained['model'], strict=False)
        
    ## Loss main classification + CF
    criterion = CustomLoss(args.loss,  args.alpha_effect)

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=130 if args.datasets=='camelyon16' else args.max_epoch,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = 0, 0, 0, 0,0,0
    opt_te_auc,opt_tea_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs  = 0., 0., 0., 0., 0.
    epoch_start = 0

    if args.fix_train_random:
        seed_torch(args.seed)

    train_time_meter = AverageMeter()
    train_loss = 0 # init train_loss
    for epoch in range(epoch_start, args.num_epoch):
        
        stop,accuracy, auc_value, precision, recall, fscore, test_loss = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch)

        if args.always_test:

            _te_accuracy, _te_auc_value, _te_precision, _te_recall, _te_fscore,_te_test_loss_log = test(args,model,test_loader,device,criterion)
            print('\rEpoch [%d/%d] Test Set , accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , test_loss: %.3f' % 
             (epoch, args.num_epoch,  _te_accuracy, _te_auc_value, _te_precision, _te_recall, _te_fscore,_te_test_loss_log ))

            

            if _te_auc_value > opt_te_auc:
                opt_te_auc = _te_auc_value
                opt_te_fs = _te_fscore
                
        if True: #not args.no_log:
            print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        (epoch, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))

        
        if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.num_epoch:
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_epoch = epoch

            # if not os.path.exists(args.model_path):
            os.makedirs(args.model_path, exist_ok = True)
            if True : 
                best_pt = {
                    'model': model.state_dict(),
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))

        # Early stopping stop
        if stop:
            print('Early stopping at epoch ', epoch)
            break

        # Train after Eval of random model / init from trained ckpt

        train_loss,start,end = train_loop(args,model,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,epoch)
        train_time_meter.update(end-start)

    # test
    best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
    info = model.load_state_dict(best_std['model'])
    print('Loading best ckp ', info)
    accuracy, auc_value, precision, recall, fscore,test_loss_log = test(args,model,test_loader,device,criterion)
 
    print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)

    if args.always_test:
        te_auc.append(opt_te_auc)
        te_fs.append(opt_te_fs)
        
    return [acs,pre,rec,fs,auc,te_auc,te_fs] # test letrics of best ckpt on test set and opt testset perf across epochs ( te_auc, te_fs)

def train_loop(args,model,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,epoch):
    start = time.time()
    loss_cls_meter = AverageMeter()
    loss_cl_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()

    train_loss_log = 0.
    model.train()

    for i, data in enumerate(loader):
        optimizer.zero_grad()

        if isinstance(data[0],(list,tuple)):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
            bag=data[0]
            batch_size=data[0][0].size(0)
        else:
            bag=data[0].to(device)  # b*n*1024
            batch_size=bag.size(0)
            
        label=data[1].to(device)
        
        with amp_autocast():
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)

            if args.model in ('clam_sb','clam_mb','dsmil'):
                train_logits_all,train_A_all,cls_loss,patch_num = model(bag,label,nn.CrossEntropyLoss())
                keep_num = patch_num
            else:
                train_logits_all, train_A_all = model(bag)
                cls_loss,patch_num,keep_num = 0.,0.,0.
            
            train_logits, train_A = train_logits_all[0], train_A_all[0] # attention squeezed of shape(N,)
            train_logits_cf, train_A_cf = train_logits_all[1], train_A_all[1]

            if args.loss == 'ce':
                logit_loss = criterion(train_logits.view(batch_size,-1),label,train_logits_cf.view(batch_size,-1),train_A,train_A_cf)
            elif args.loss == 'bce':
                logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2),train_logits_cf.view(batch_size,-1),train_A,train_A_cf)

        train_loss = args.cls_alpha * logit_loss +  cls_loss*args.aux_alpha

        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            train_loss.backward()
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()

        loss_cls_meter.update(logit_loss,1)
        loss_cl_meter.update(cls_loss,1)
        patch_num_meter.update(patch_num,1)
        keep_num_meter.update(keep_num,1)

        if i % args.log_iter == 0 or i == len(loader)-1:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('cls_loss',loss_cls_meter.avg),
                ('lr',lr),
                ('cl_loss',loss_cl_meter.avg),
                ('patch_num',patch_num_meter.avg),
                ('keep_num',keep_num_meter.avg),
            ])
            if True: #not args.no_log:
                print('[{}/{}] logit_loss:{}, cls_loss:{},  patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,patch_num_meter.avg, keep_num_meter.avg))
            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log,start,end

def val_loop(args,model,loader,device,criterion,early_stopping,epoch):
    model.eval()
    loss_cls_meter = AverageMeter()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []
    # pred= []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)

            if args.model == 'dsmil':
                test_logits_all,test_A_all,_ = model(bag)
            else:
                test_logits_all, test_A_all = model(bag)
            test_logits, test_A = test_logits_all[0], test_A_all[0] # attention squeezed of shape(N,)
            test_logits_cf, test_A_cf = test_logits_all[1], test_A_all[1]

            if args.loss == 'ce':
                test_loss = criterion(test_logits.view(batch_size,-1),label,test_logits_cf.view(batch_size,-1),test_A,test_A_cf)
                if batch_size > 1:
                    bag_logit.extend(torch.softmax(test_logits.view(batch_size,-1),dim=-1)[:,1].cpu().squeeze().numpy())
                else:
                    bag_logit.append(torch.softmax(test_logits.view(batch_size,-1),dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':

                test_loss = criterion(test_logits[0].view(batch_size,-1),label.view(batch_size,-1).float(),test_logits_cf.view(batch_size,-1),test_A,test_A_cf)

                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            loss_cls_meter.update(test_loss,1)
    
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit, not args.datasets.lower() == 'camelyon16')
    
    # early stop
    if early_stopping is not None:
        early_stopping(epoch,-auc_value,model)
        stop = early_stopping.early_stop
    else:
        stop = False
    return stop,accuracy, auc_value, precision, recall, fscore,loss_cls_meter.avg

def test(args,model,loader,device,criterion):
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())
                
            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)

            if args.model == 'dsmil':
                test_logits_all,test_A_all,_ = model(bag)
            else:
                test_logits_all, test_A_all = model(bag)

            test_logits, test_A = test_logits_all[0], test_A_all[0] # attention squeezed of shape(N,)
            test_logits_cf, test_A_cf = test_logits_all[1], test_A_all[1]

            if args.loss == 'ce':
              
                test_loss = criterion(test_logits.view(batch_size,-1),label,test_logits_cf.view(batch_size,-1),test_A,test_A_cf)
                if batch_size > 1:
                    bag_logit.extend(torch.softmax(test_logits.view(batch_size,-1),dim=-1)[:,1].cpu().squeeze().numpy())
                else:
                    bag_logit.append(torch.softmax(test_logits.view(batch_size,-1),dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
               
                test_loss = criterion(test_logits.view(batch_size,-1),label.view(1,-1).float(),test_logits_cf.view(batch_size,-1),test_A,test_A_cf)
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit, not args.datasets.lower() == 'camelyon16')
    test_loss_log = test_loss_log/len(loader)

    return accuracy, auc_value, precision, recall, fscore,test_loss_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga, luad_tp53]')
    parser.add_argument('--dataset_root', default='/data/xxx/TransMIL', type=str, help='Dataset root path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio') #by default we validate on the test set
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    parser.add_argument('--tcga_sub', default='nsclc', type=str, help='[nsclc,brca]')
    
    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--aux_alpha', default=1.0, type=float, help='Auxiliary loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=100, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=70, type=int, help='Number of max training epochs in the earlystopping [70]')
    parser.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='attmil', type=str, help='Model name')
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')

    # Model 
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    # CF
    parser.add_argument('--alpha_effect', default=0.2, type=float, help='Classification Loss [ce, bce]')
    parser.add_argument('--init_ckpt', default='none',type=str, help='path of trained ckpt directory or pt file')
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

    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    if 'clam_sb' in args.model :
        args.cls_alpha= .7
        args.aux_alpha = .3
    elif 'clam_mb' in args.model:
        args.cls_alpha= .7
        args.aux_alpha = .3
    elif 'dsmil' in args.model:
        args.cls_alpha = 0.5
        args.aux_alpha = 0.5

    if args.datasets.lower() == 'camelyon16':
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets.lower() == 'tcga':
        # args.num_workers = 0
        args.fix_loader_random = True
        args.fix_train_random = True
        # args.always_test = True

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
