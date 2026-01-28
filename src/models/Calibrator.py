import torch
import torch.nn as nn
import torch.nn.functional as F

cate_bin_size = (5,3604778,1951876,432750,14,19,2638598,1962400, 11123, 397159, 578622,3,622,107,8009,30685,104,10,3,2,9,7) #one_code_each only multi user

_EPSILON = 1e-8

class Dice(nn.Module):
    def __init__(self, units, eps=1e-4):
        super(Dice, self).__init__()
        self.bn = nn.LayerNorm(units, eps=eps, elementwise_affine=False)
        self.alpha = nn.Parameter(torch.ones(1, units) * -0.25)

    def forward(self, x):
        normed = self.bn(x)
        p = torch.sigmoid(normed)
        return p * x + (1 - p) * self.alpha * x

class CalibratorWithTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cate_features = len(cate_bin_size)
        self.all_bin_sizes = list(cate_bin_size)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])
        input_dim = args.embed_dim * self.cate_features
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.dice1 = Dice(512)
        self.fc2 = nn.Linear(512,256)
        self.dice2 = Dice(256)
        self.fc3 = nn.Linear(256, 128)  
        self.dice3 = Dice(128)
        self.fc4 = nn.Linear(128, 64)
        self.dice4 = Dice(64)
        self.fc5 = nn.Linear(64+2, 1)  
        self.softplus = nn.Softplus()
        
    def forward(self, x, delta_t, k):
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        x = self.dice1(self.fc1(x_embed))
        x = self.dice2(self.fc2(x))
        x = self.dice3(self.fc3(x))
        x = self.dice4(self.fc4(x))
        x = torch.cat([x, delta_t.unsqueeze(1), k.unsqueeze(1)], dim=-1)
        x = self.softplus(self.fc5(x))
        return x

