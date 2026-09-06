import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

class AttentionGated_CF(nn.Module):
    def __init__(self,input_dim,act='relu',n_classes=2,bias=False,dropout=False):
        super(AttentionGated_CF, self).__init__()
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
        self.attention_c_cf = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    
    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A_raw = a.mul(b)
        
        A = self.attention_c(A_raw)
        A_cf = self.attention_c_cf(A_raw)

        A = torch.transpose(A, -1, -2)  # KxN
        A_cf = torch.transpose(A_cf, -1, -2)

        A_ori = torch.cat((A,A_cf),0)
        
        A = F.softmax(A, dim=-1)  # softmax over N
        x_f = torch.matmul(A,x)

        A_cf = F.softmax(A_cf, dim=-1)  # softmax over N
        x_cf = torch.matmul(A_cf,x)

        Y_prob = self.classifier(x_f)
        Y_prob_cf = self.classifier(x_cf)

        Y = torch.cat((Y_prob,Y_prob_cf),0)
        

        return Y, A_ori
    
    def forward_diff(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A_raw = a.mul(b)
        
        A = self.attention_c(A_raw)
        A_cf = self.attention_c_cf(A_raw)
        A_diff = A - A_cf

        A = torch.transpose(A, -1, -2)  # KxN
        A_cf = torch.transpose(A_cf, -1, -2)
        A_diff = torch.transpose(A_diff, -1, -2)

        A_ori = torch.cat((A,A_cf),0)
        
        A = F.softmax(A, dim=-1)  # softmax over N
        x_f = torch.matmul(A,x)

        A_cf = F.softmax(A_cf, dim=-1)  # softmax over N
        x_cf = torch.matmul(A_cf,x)

        A_diff = F.softmax(A_diff, dim=-1)
        x_diff = torch.matmul(A_diff,x)

        Y_prob = self.classifier(x_diff)
        Y_prob_cf = self.classifier(x_cf)

        Y = torch.cat((Y_prob,Y_prob_cf),0)
        

        return Y, A_ori

    def forward_eval(self, x):
        x = self.feature(x.squeeze(0))
        a = self.attention_a(x)
        b = self.attention_b(x)
        A_raw = a.mul(b)
        
        A = self.attention_c(A_raw)
        A_cf = self.attention_c_cf(A_raw)

        A = torch.transpose(A, -1, -2)  # KxN
        A_cf = torch.transpose(A_cf, -1, -2)

        A_ori = torch.cat((A,A_cf),0)
        
        A = F.softmax(A, dim=-1)  # softmax over N
        x_f = torch.matmul(A,x)

        A_cf = F.softmax(A_cf, dim=-1)  # softmax over N
        x_cf = torch.matmul(A_cf,x)

        Y_prob = self.classifier(x_f)
        Y_prob_cf = self.classifier(x_cf)

        Y = torch.cat((Y_prob,Y_prob_cf),0)
        

        return Y, A_ori


class AttentionGated_CAL(nn.Module):
    def __init__(self,input_dim,act='relu',n_classes=2,bias=False,dropout=False):
        super(AttentionGated_CAL, self).__init__()
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
        #self.attention_c_cf = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    
    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A_raw = a.mul(b)
        
        A = self.attention_c(A_raw)
        
        
        #A_cf = self.attention_c_cf(A_raw)

        # === RANDOM ATTENTION SAMPLE ===
        N = x.size(0)      # number of patches
        K = 1              # number of attention heads

        A_cf = torch.rand(K, N, device=x.device)    # random attention before softmax
                      # normalize
        

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = torch.cat((A,A_cf),0)
        
        A = F.softmax(A, dim=-1)  # softmax over N
        x_f = torch.matmul(A,x)

        A_cf = F.softmax(A_cf, dim=-1)  # softmax over N
        x_cf = torch.matmul(A_cf,x)

        Y_prob = self.classifier(x_f)
        Y_prob_cf = self.classifier(x_cf)

        Y = torch.cat((Y_prob,Y_prob_cf),0)
        

        return Y, A_ori
    

    def forward_eval(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A_raw = a.mul(b)
        
        A = self.attention_c(A_raw)
        
        

        # === RANDOM ATTENTION SAMPLE ===
        N = x.size(0)      # number of patches
        K = 1              # number of attention heads

        A_cf = torch.rand(K, N, device=x.device)    # random attention before softmax
                      # normalize
        

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = torch.cat((A,A_cf),0)
        
        A = F.softmax(A, dim=-1)  # softmax over N
        x_f = torch.matmul(A,x)

        A_cf = F.softmax(A_cf, dim=-1)  # softmax over N
        x_cf = torch.matmul(A_cf,x)

        Y_prob = self.classifier(x_f)
        Y_prob_cf = self.classifier(x_cf)

        Y = torch.cat((Y_prob,Y_prob_cf),0)
        
        return Y, A_ori

class DAttention_CF(nn.Module):
    def __init__(self,input_dim,n_classes,dropout,act):
        super(DAttention_CF, self).__init__()
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

        self.attention_cf = nn.Sequential(
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
        A_cf = self.attention_cf(feature)

    
        A = torch.transpose(A, -1, -2)  # KxN
        A_cf = torch.transpose(A_cf, -1, -2)  # KxN
        
        A_ori = torch.cat((A,A_cf),dim=0)
        
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        Y_prob = self.classifier(M)

        A_cf = F.softmax(A_cf, dim=-1)  # softmax over N
        M_cf = torch.mm(A_cf, feature)  # KxL
        Y_prob_cf = self.classifier(M_cf)

        Y = torch.cat((Y_prob,Y_prob_cf),dim=0)

        return Y, A_ori
        
    def forward_eval(self, x, return_attn=True,no_norm=True):
        feature = self.feature(x)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_cf = self.attention_cf(feature)

    
        A = torch.transpose(A, -1, -2)  # KxN
        A_cf = torch.transpose(A_cf, -1, -2)  # KxN
        
        A_ori = torch.cat((A,A_cf),dim=0)
        
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        Y_prob = self.classifier(M)

        A_cf = F.softmax(A_cf, dim=-1)  # softmax over N
        M_cf = torch.mm(A_cf, feature)  # KxL
        Y_prob_cf = self.classifier(M_cf)

        Y = torch.cat((Y_prob,Y_prob_cf),dim=0)

        return Y, A_ori


