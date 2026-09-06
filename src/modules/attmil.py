import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # ref from meituan
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class AttentionGated(nn.Module):
    def __init__(self,input_dim,act='relu',n_classes=2,bias=False,dropout=False):
        super(AttentionGated, self).__init__()
        self.L = 512
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(input_dim, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
       
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        Y_prob = self.classifier(x)

        return Y_prob

    def forward_eval(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A_ori = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A_ori, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        Y_prob = self.classifier(x)

        return Y_prob, A_ori

class DAttention(nn.Module):
    def __init__(self,input_dim,n_classes,dropout,act):
        super(DAttention, self).__init__()
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1
        self.feature = [nn.Linear(input_dim, 512)]
        
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]
      
        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )
        
        self.apply(initialize_weights)
    def forward(self, x, return_attn=False,no_norm=False):
        feature = self.feature(x)

        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        Y_prob = self.classifier(M)

        if return_attn:
            if no_norm:
                return Y_prob,A_ori
            else:
                return Y_prob,A
        else:
            return Y_prob
        
    def forward_eval(self, x, return_attn=True,no_norm=True):
        feature = self.feature(x)

        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        Y_prob = self.classifier(M)

        if return_attn:
            if no_norm:
                return Y_prob,A_ori
            else:
                return Y_prob,A
        else:
            return Y_prob



class Inst_AttentionGated(nn.Module):
    def __init__(self,input_dim,act='relu',n_classes=2,bias=False,dropout=False):
        super(Inst_AttentionGated, self).__init__()
        self.L = 512
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(input_dim, self.L)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
       
        self.feature = nn.Sequential(*self.feature)
        self.bag_classifier = nn.Linear(self.L, n_classes)

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N

        Y_inst = self.bag_classifier(x) #N x n_classes
        Y_prob = torch.mm(A, Y_inst)  # Kxn_classes

        return Y_prob

    def forward_eval(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A_ori = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A_ori, dim=-1)  # softmax over N

        Y_inst = self.bag_classifier(x) #N x n_classes
        Y_prob = torch.mm(A, Y_inst)  # Kxn_classes

        return Y_prob, A_ori



class Add_AttentionGated(nn.Module):
    def __init__(self,input_dim,act='relu',n_classes=2,bias=False,dropout=False):
        super(Add_AttentionGated, self).__init__()
        self.L = 512
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(input_dim, self.L)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
       
        self.feature = nn.Sequential(*self.feature)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N

        A = torch.transpose(A, -1, -2)
        x_inst = A*x  # Kxn_classes
        Y_prob = self.classifier(x_inst)
        Y_prob = Y_prob.sum(axis=0, keepdims=True)
        
        return Y_prob

    def forward_eval(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A_ori = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A_ori, dim=-1)  # softmax over N
        

        A = torch.transpose(A, -1, -2)
        x_inst = A*x  # Kxn_classes
        Y_prob = self.classifier(x_inst)
        Y_prob = Y_prob.sum(axis=0, keepdims=True)
     
        return Y_prob, A_ori


class IBMIL_AttentionGated(nn.Module):
    def __init__(self,input_dim,act='relu',n_classes=2,bias=False,dropout=False, confounder_path=False, confounder_dim=128, confounder_merge='cat'):
        super(IBMIL_AttentionGated, self).__init__()
        self.L = 512
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(input_dim, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
       
        self.feature = nn.Sequential(*self.feature)


        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        ## ibmil 
        self.confounder_merge = confounder_merge
        assert confounder_merge in ['cat', 'add', 'sub']
        


        self.confounder_path=None
        if confounder_path: 
            print('deconfounding')
            self.confounder_path = confounder_path
            conf_list = []
            for i in confounder_path:
                conf_list.append(torch.from_numpy(np.load(i)).view(-1,512).float()) # hard coded here
            conf_tensor = torch.cat(conf_list, 0) 
            conf_tensor_dim = conf_tensor.shape[-1]

            assert self.L == conf_tensor_dim # == 512
            
            
            self.register_buffer("confounder_feat",conf_tensor)

            joint_space_dim = confounder_dim
            dropout_v = 0.5
       
            self.W_q = nn.Linear(self.L , joint_space_dim)
            self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            if confounder_merge == 'cat':
                self.classifier =  nn.Linear(self.L*self.K+conf_tensor_dim, n_classes)
          
            self.dropout = nn.Dropout(dropout_v)

        
        
        # self.apply(initialize_weights)
    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.matmul(A,x)

        if self.confounder_path:
            device = M.device
         
            bag_q = self.W_q(M)
            conf_k = self.W_k(self.confounder_feat)
            deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
            deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            conf_feats = torch.mm(deconf_A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
            if self.confounder_merge == 'cat':
                M = torch.cat((M,conf_feats),dim=1)
            elif self.confounder_merge == 'add':
                M = M + conf_feats
            elif self.confounder_merge == 'sub':
                M = M - conf_feats



        Y_prob = self.classifier(M)

        return Y_prob

    def forward_eval(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A_ori = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A_ori, dim=-1)  # softmax over N
        M = torch.matmul(A,x)

        if self.confounder_path:
            device = M.device
         
            bag_q = self.W_q(M)
            conf_k = self.W_k(self.confounder_feat)
            deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
            deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            conf_feats = torch.mm(deconf_A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
            if self.confounder_merge == 'cat':
                M = torch.cat((M,conf_feats),dim=1)
            elif self.confounder_merge == 'add':
                M = M + conf_feats
            elif self.confounder_merge == 'sub':
                M = M - conf_feats

        Y_prob = self.classifier(M)

        return Y_prob, A_ori

