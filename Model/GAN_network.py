import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self,features):
        self.layer_num = len(features)-1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
                                         nn.Linear(features[-1], features[-2]),
                                         nn.Linear(features[-2], features[-2]),
                                         nn.Linear(features[-2], 1))

    def forward(self, f):
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out) # (B, 1)

        return out


class Discriminator_avg_pooling(nn.Module):
    def __init__(self,features,class_num):
        self.layer_num = len(features)-1
        super(Discriminator_avg_pooling, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))
        self.linear_pc = nn.Conv1d(3,features[0], kernel_size=1, stride=1)
        self.linear_class = nn.Conv1d(class_num,features[0], kernel_size=1, stride=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
                                         nn.Linear(features[-1], features[-2]),
                                         nn.Linear(features[-2], features[-2]),
                                         nn.Linear(features[-2], 1))

    def forward(self, f):
        feat = f.transpose(1,2) # B * K * N
        vertex_num = feat.size(2)

        pc = feat[:,:3,:]
        classes = feat[:,3:,:]
        pc = self.linear_pc(pc)
        classes = self.linear_class(classes)
        feat = pc + classes

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.avg_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out) # (B, 1)

        return out


class Generator(nn.Module):
    def __init__(self, Net_0, Net_1, K = 32):
        self.pointcloud = None
        super(Generator, self).__init__()
        self.K = K
        self.Net_0 = Net_0
        self.Net_1 = Net_1
        self.pc_0 = None
        self.pc_1 = None

    def forward(self,z):
        """
            Z : (b , N)
            Z2 : (b , K , N)
        """
        shape = z.shape
        output_0 = self.Net_0(z) # (b , K , Feat)

        z2 = z.view(shape[0],1,shape[1])
        z2 = z2.expand(shape[0],self.K,shape[1])

        output_1 = self.Net_1(output_0,z2) # 2048 * 3
        
        self.pc_0 = output_0
        self.pc_1 = output_1

        return output_0,output_1
    
    def get_pcs(self):
        return self.pc_0[-1],self.pc_1[-1]

