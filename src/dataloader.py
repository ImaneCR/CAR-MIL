import os
import csv
import torch
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import h5py


FAILED_SLIDES = ['TCGA-05-5425-01Z-00-DX1.85865B2F-4888-43DD-A501-458BEFCF832B',
                'TCGA-05-4384-01Z-00-DX1.CA68BF29-BBE3-4C8E-B48B-554431A9EE13',
                'TCGA-05-5428-01Z-00-DX1.8018AD62-F1CE-4BFF-8EFD-3F2D4513FC11',
                'TCGA-05-4410-01Z-00-DX1.E5B66334-4949-4F45-9200-296B1A2F1AD5',
                'TCGA-05-5423-01Z-00-DX1.CCCF5FDB-ACAD-4D9D-80DF-556F0D6284AF',
                'TCGA-05-4390-01Z-00-DX1.858E64DF-DD3E-4F43-B7C1-CE35B33F1C90',
                'TCGA-05-5420-01Z-00-DX1.8C253A99-44FD-48B6-AF31-D808CCB7DB1E',
                'TCGA-55-8207-01Z-00-DX1.2dafc442-f927-4b0d-b197-cc8c5f86d0fc',
                'TCGA-05-5715-01Z-00-DX1.D3F0A1FA-2507-45FF-823F-F9981E62BB4C',
                'TCGA-05-4425-01Z-00-DX1.82B093EE-49BC-4FD9-91AC-4CC89944309D',
                'TCGA-05-5429-01Z-00-DX1.20729065-FADA-4E43-98D7-AFA5FB4A0447']

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def get_patient_label(csv_file):
    patients_list=[]
    labels_list=[]
    label_file = readCSV(csv_file)
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

def get_patient_label_c16(csv_file):
    patients_list=[]
    labels_list=[]
    label_file_df = pd.read_csv(csv_file)
    patients_list = list(label_file_df['slide_id'])
    labels_list = list(label_file_df['label'])

    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

def get_patient_label_luad(csv_file, exclude_slides = FAILED_SLIDES):
    patients_list=[]
    labels_list=[]
    label_file_df_raw = pd.read_csv(csv_file)

    # Exclude failed slides feat:

    label_file_df = label_file_df_raw[~label_file_df_raw.slide_id.isin(FAILED_SLIDES)]
    patients_list = list(label_file_df['slide_id'])
    labels_list = list(label_file_df['TP53'])

    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    #return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)
    return np.array(patients_list), np.array(labels_list)

def get_patient_label_bracs(csv_file, config ):
    patients_list=[]
    labels_list=[]
    label_file_df = pd.read_csv(csv_file)

    if config == '3cl':
        col_label = 'label_3classes'
    else: #7 classes
        col_label = 'label_7classes'
    
    patients_list = list(label_file_df['slide_id'])
    labels_list = list(label_file_df[col_label])
    split_list = list(label_file_df['split'])

    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    #return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

    train_patients_list = list(label_file_df[label_file_df.split=='train']['slide_id'])
    train_labels_list = list(label_file_df[label_file_df.split=='train'][col_label])
    
    test_patients_list = list(label_file_df[label_file_df.split=='test']['slide_id'])
    test_labels_list = list(label_file_df[label_file_df.split=='test'][col_label])
    
    val_patients_list = list(label_file_df[label_file_df.split=='val']['slide_id'])
    val_labels_list = list(label_file_df[label_file_df.split=='val'][col_label])


    #return np.array(patients_list), np.array(labels_list), np.array(split_list)

    return np.array(train_patients_list), np.array(train_labels_list), np.array(test_patients_list), np.array(test_labels_list),np.array(val_patients_list), np.array(val_labels_list)




def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=True):
    """
    dataset split: split the full_list randomly into two sublist (val-set and train-set) based on the ratio
    :param full_list: 
    :param ratio:     
    :param shuffle:  
    """
    # select the val-set based on the label ratio
    if label_balance_val and label is not None:
        _label = label[full_list]
        _label_uni = np.unique(_label)
        sublist_1 = []
        sublist_2 = []

        for _l in _label_uni:
            _list = full_list[_label == _l]
            n_total = len(_list)
            offset = int(n_total * ratio)
            if shuffle:
                random.shuffle(_list)
            sublist_1.extend(_list[:offset])
            sublist_2.extend(_list[offset:])
        val_set, train_set = sublist_1, sublist_2
    else:
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        val_set = full_list[:offset]
        train_set = full_list[offset:]

    return val_set, train_set



def get_kflod(k, patients_array, labels_array,val_ratio=False,label_balance_val=True):
    if k > 1:
        skf = StratifiedKFold(n_splits=k)
    else:
        raise NotImplementedError
    train_patients_list = []
    train_labels_list = []
    test_patients_list = []
    test_labels_list = []
    val_patients_list = []
    val_labels_list = []
    for train_index, test_index in skf.split(patients_array, labels_array):
        if val_ratio != 0.:
            val_index,train_index = data_split(train_index,val_ratio,True,labels_array,label_balance_val)
            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val, y_val = [],[]
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)
        
    # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
    return np.array(train_patients_list,dtype=object), np.array(train_labels_list,dtype=object), np.array(test_patients_list,dtype=object), np.array(test_labels_list,dtype=object),np.array(val_patients_list,dtype=object), np.array(val_labels_list,dtype=object)

def get_tcga_parser(root,cls_name,mini=False):
        x = []
        y = []

        for idx,_cls in enumerate(cls_name):
            _dir = 'mini_pt' if mini else 'pt_files'
            _files = os.listdir(os.path.join(root,_cls,'features',_dir))
            _files = [os.path.join(os.path.join(root,_cls,'features',_dir,_files[i])) for i in range(len(_files))]
            x.extend(_files)
            y.extend([idx for i in range(len(_files))])
            
        return np.array(x).flatten(),np.array(y).flatten()

class TCGADataset(Dataset):
    
    def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=False,keep_same_psize=0,is_train=False,_type='nsclc'):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset, self).__init__()

        self.patient_name = file_name
        self.patient_label = file_label
        self.max_patch = max_patch
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'h5_files')) if keep_same_psize else os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.slide_label = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
                else:
                    self.slide_name.append(_sides[_idx])
                self.slide_label.append(self.patient_label[i])
                
        if _type.lower() == 'nsclc':
            self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]
        elif _type.lower() == 'brca':
            self.slide_label = [ 0 if _l == 'IDC' else 1 for _l in self.slide_label]

    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        label = self.slide_label[idx]

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path))

        # features: [N, C]
        if self.max_patch is not None and self.max_patch > 0 and features.size(0) > self.max_patch:
            if self.is_train:
                # stochastic subset each epoch
                perm = torch.randperm(features.size(0))[:self.max_patch]
            else:
                # deterministic subset for eval (stable metrics)
                g = torch.Generator()
                g.manual_seed(idx)  # or hash(file_path) if you want
                perm = torch.randperm(features.size(0), generator=g)[:self.max_patch]
            features = features[perm]
        return features , int(label)

class C16Dataset(Dataset):

    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [ 0 if _l == 'normal' else 1 for _l in self.slide_label] #[int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        # if persistence:
        #     self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"h5_files")

            file_path = os.path.join(dir_path, self.file_name[idx]+'.h5')
            with h5py.File(file_path,'r') as slide:
                np_features = slide['features'][:]
    
            features = torch.tensor(np_features)

        label = int(self.slide_label[idx])


        return features , label

class C16Dataset_TEST(Dataset):

    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [ 0 if _l == 'normal' else 1 for _l in self.slide_label] #[int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        # if persistence:
        #     self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"h5_files")

            file_path = os.path.join(dir_path, self.file_name[idx]+'.h5')
            with h5py.File(file_path,'r') as slide:
                np_features = slide['features'][:]
    
            features = torch.tensor(np_features)

        label = int(self.slide_label[idx])


        return features , label
    
class TCGADataset_LUAD(Dataset):
    
    def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=False,keep_same_psize=0,is_train=False,_type='nsclc'):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset_LUAD, self).__init__()

        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.max_patch = max_patch
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        # if persistence:
        #     self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"h5_files")

            file_path = os.path.join(dir_path, self.file_name[idx]+'.h5')
            with h5py.File(file_path,'r') as slide:
                np_features = slide['features'][:]    
            features = torch.tensor(np_features)

        label = int(self.slide_label[idx])

        if self.max_patch is not None and self.max_patch > 0 and features.size(0) > self.max_patch:
            if self.is_train:
                # stochastic subset each epoch
                perm = torch.randperm(features.size(0))[:self.max_patch]
            else:
                # deterministic subset for eval (stable metrics)
                g = torch.Generator()
                g.manual_seed(idx)  # or hash(file_path) if you want
                perm = torch.randperm(features.size(0), generator=g)[:self.max_patch]
            features = features[perm]

        return features , label


class TCGADataset_BRACS(Dataset):
    
    def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset_BRACS, self).__init__()

        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        # if persistence:
        #     self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = self.root #os.path.join(self.root,"h5_files")

            file_path = os.path.join(dir_path, self.file_name[idx]+'.h5')
            with h5py.File(file_path,'r') as slide:
                np_features = slide['features'][:]    
            features = torch.tensor(np_features)

        label = int(self.slide_label[idx])
        return features , label
