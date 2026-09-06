import torch
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention
import torch


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
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x,return_attn=False):
        # x = x + self.attn(self.norm(x))
        # return x
        if return_attn :
            attn_out, full_att = self.attn(self.norm(x), return_attn=True)
            x = x + attn_out
            return x, full_att   # full_att: [B, heads, T,T]
        else:
            attn_out = self.attn(self.norm(x), return_attn=False)
            x = x + attn_out
            return x   

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_dim, n_classes,dropout,act):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = [nn.Linear(input_dim, 512)]

        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout:
            self._fc1 += [nn.Dropout(0.25)]

        self._fc1 = nn.Sequential(*self._fc1)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, x):

        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h,return_attn=False) #[B, N +1, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N +1, 512]
        
        #---->Translayer x2
        h  = self.layer2(h,return_attn=False) #[B, N +1, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
   
        return logits

    def forward_eval(self, x, layer=None):
        n = x.shape[1]
        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        before_pad = h.shape[-2]

        #---->Translayer x1
        h,att1 = self.layer1(h,return_attn=True) #[B, N +1, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N +1, 512]
        
        #---->Translayer x2
        h , att2 = self.layer2(h,return_attn=True) #[B, N +1, 512]
        
        #---->cls_token
        h = self.norm(h)[:,0]
        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        
        ## Attention maps
        n_attention_tokens = att1.shape[-2]
        attn_pad_tokens = n_attention_tokens - before_pad

        if layer == None: # return rollout of attention
            result = torch.eye(n_attention_tokens).to(h.device)
            with torch.no_grad():
                for att_map in [att1, att2]:
                    att_fused = att_map.mean(axis=1)
                    ## set some values to discard_ratio
                    # flat = att_fused.view(att_fused.size(0), -1)
                    # _, indices = flat.topk(discard_ratio, -1, False)
                    # indices = indices[indices != attn_pad_tokens]
                    # flat[0, indices] = 0

                    I = torch.eye(att_fused.size(-1)).to(h.device)
                    a = (att_fused + 1.0 * I) / 2
                    a = a / a.sum(dim=-1)
                    result = torch.matmul(a, result)

        
        elif layer == 2:
            result = att2.mean(axis=1) # avg pooling of heads
        

        WSI_attn = result[0,attn_pad_tokens,attn_pad_tokens+1:attn_pad_tokens + 1 + n]
        
        return logits, WSI_attn # attention here is not softmaxed



class TransMIL_CF(nn.Module):
    def __init__(self, input_dim=1024, n_classes=2, dropout=0.25, act='relu'):
        super(TransMIL_CF, self).__init__()
        self.pos_layer = PPEG(dim=512)

        self._fc1 = [nn.Linear(input_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(0.25)]
        self._fc1 = nn.Sequential(*self._fc1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)

        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.layer2_cf = TransLayer(dim=512)   # <-- CF branch

        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, n_classes)

        self.apply(initialize_weights)

    def forward(self, x):
        # --- embed patches
        h0 = x.float()
        h0 = self._fc1(h0)
        if len(h0.size()) == 2:
            h0 = h0.unsqueeze(0) # B n dim
        n = h0.shape[1]

        # --- pad to square (same as your code)
        H = h0.shape[1]
        _H = _W = int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h0 = torch.cat([h0, h0[:, :add_length, :]], dim=1)
        

        # --- add CLS
        B = h0.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h0.device)
        h0 = torch.cat((cls_tokens, h0), dim=1)  # [B, 1+Npad, 512]
        before_pad = h0.shape[-2]

        # --- shared block
        h1 = self.layer1(h0,return_attn=False)
        h1 = self.pos_layer(h1, _H, _W)

        h1_cf = h1.clone()
        # --- original path
        h2, att2 = self.layer2(h1,return_attn=True)     # att2: [B, heads, Npad]
        cls_ori = self.norm(h2)[:, 0]
        Y_ori = self._fc2(cls_ori)     # [B, C]


        # --- counterfactual path
        h2cf, att2cf = self.layer2_cf(h1_cf,return_attn=True)
        cls_cf = self.norm(h2cf)[:, 0]
        Y_cf = self._fc2(cls_cf)       # [B, C]  (shared classifier like your example)


        n_attention_tokens = att2.shape[-2]
        attn_pad_tokens = n_attention_tokens - before_pad

        # --- build A_ori (raw, before softmax), match your A/A_cf cat
        # your NystromAttention already returns per-head scores; aggregate heads to get [B, Npad]
        att2_fused = att2.mean(dim=1)       
        cls_att2 = att2_fused[0,attn_pad_tokens,attn_pad_tokens+1:attn_pad_tokens + 1 + n]

        att2cf_fused = att2cf.mean(dim=1)       
        cls_att2cf = att2cf_fused[0,attn_pad_tokens,attn_pad_tokens+1:attn_pad_tokens + 1 + n]

        A_ori = torch.cat([cls_att2, cls_att2cf], dim=0)   # [2B, n]
        Y = torch.cat([Y_ori, Y_cf], dim=0)           # [2B, C]
       
        return Y, A_ori
    
    
    def forward_eval(self, x,layer=None):
        # --- embed patches
        h0 = x.float()
        h0 = self._fc1(h0)
        if len(h0.size()) == 2:
            h0 = h0.unsqueeze(0) # B n dim
        n = h0.shape[1]

        # --- pad to square (same as your code)
        H = h0.shape[1]
        _H = _W = int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h0 = torch.cat([h0, h0[:, :add_length, :]], dim=1)
        

        # --- add CLS
        B = h0.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h0.device)
        h0 = torch.cat((cls_tokens, h0), dim=1)  # [B, 1+Npad, 512]
        before_pad = h0.shape[-2]

        # --- shared block
        h1, att1 = self.layer1(h0,return_attn=True)
        h1 = self.pos_layer(h1, _H, _W)

        h1_cf = h1.clone()
        # --- original path
        h2, att2 = self.layer2(h1,return_attn=True)     # att2: [B, heads, Npad]
        cls_ori = self.norm(h2)[:, 0]
        Y_ori = self._fc2(cls_ori)     # [B, C]


        # --- counterfactual path
        h2cf, att2cf = self.layer2_cf(h1_cf,return_attn=True)
        cls_cf = self.norm(h2cf)[:, 0]
        Y_cf = self._fc2(cls_cf)       # [B, C]  (shared classifier like your example)


        n_attention_tokens = att2.shape[-2]
        attn_pad_tokens = n_attention_tokens - before_pad

        # --- build A_ori (raw, before softmax), match your A/A_cf cat
        # your NystromAttention already returns per-head scores; aggregate heads to get [B, Npad]
        
        att2cf_fused = att2cf.mean(dim=1)       
        cls_att2cf = att2cf_fused[:,attn_pad_tokens,attn_pad_tokens+1:attn_pad_tokens + 1 + n]


        if layer == None:
            att2_fused = torch.eye(n_attention_tokens).to(h0.device)
            with torch.no_grad():
                for att_map in [att1, att2]:
                    att_fused = att_map.mean(axis=1)
                    I = torch.eye(att_fused.size(-1)).to(h0.device)
                    a = (att_fused + 1.0 * I) / 2
                    a = a / a.sum(dim=-1)
                    att2_fused = torch.matmul(a, att2_fused)

        elif layer == 2:
            att2_fused = att2.mean(dim=1)

        
        cls_att2 = att2_fused[:,attn_pad_tokens,attn_pad_tokens+1:attn_pad_tokens + 1 + n]

        WSI_attn = torch.cat([cls_att2, cls_att2cf], dim=0)   # [2B, n]
        Y = torch.cat([Y_ori, Y_cf], dim=0)           # [2B, C]

        return Y, WSI_attn

        

        
if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024))
    model = TransMIL(n_classes=2,dropout=False,act='relu')
    for k, v in model.state_dict().items():
        print(k)
    # print(model.eval())
    # results_dict = model(data = data)
    # print(results_dict)