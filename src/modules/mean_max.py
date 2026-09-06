import torch.nn as nn

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
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


class MeanMIL(nn.Module):
    def __init__(self,input_dim,n_classes=2,dropout=True,act='relu'):
        super(MeanMIL, self).__init__()

        head = [nn.Linear(input_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]

        head += [nn.Linear(512,n_classes)]
        
        self.head = nn.Sequential(*head)


        self.apply(initialize_weights)

    def forward(self,x):

        x = self.head(x).mean(axis=1)
        return x

class Embed_MeanMIL(nn.Module):
    def __init__(self,input_dim,n_classes=2,dropout=True,act='relu'):
        super(Embed_MeanMIL, self).__init__()

        head = [nn.Linear(input_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
        
        self.head = nn.Sequential(*head)
        self.classifier = nn.Linear(512,n_classes)

        self.apply(initialize_weights)

    def forward(self,x):
        x = self.head(x).mean(axis=1)
        x = self.classifier(x)
        return x


class MaxMIL(nn.Module):
    def __init__(self,input_dim,n_classes=2,dropout=True,act='relu'):
        super(MaxMIL, self).__init__()

        head = [nn.Linear(input_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
     
        head += [nn.Linear(512,n_classes)]
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    def forward(self,x):
        x,_ = self.head(x).max(axis=1)
        return x

class Embed_MaxMIL(nn.Module):
    def __init__(self,input_dim,n_classes=2,dropout=True,act='relu'):
        super(Embed_MaxMIL, self).__init__()

        head = [nn.Linear(input_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
     
        self.head = nn.Sequential(*head)
        self.classifier = nn.Linear(512,n_classes)

        self.apply(initialize_weights)

    def forward(self,x):
        x,_ = self.head(x).max(axis=1)
        x = self.classifier(x)
        return x