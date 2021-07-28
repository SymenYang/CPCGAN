import torch
import torch.nn as nn
import torch.nn.functional as F

class PC_MLP(nn.Module):
    def __init__(self, feat_channels, K, class_num = 0, norm = False):
        super(PC_MLP, self).__init__()
        self.layer_num = len(feat_channels) - 1
        self.class_num = class_num
        self.K = K
        self.norm = norm

        self.fc_layers = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        for i in range(self.layer_num):
            self.fc_layers.append(
                nn.Linear(feat_channels[i],feat_channels[i + 1])
            )
        if self.norm:
            for i in range(self.layer_num):
                self.norms.append(
                    nn.LayerNorm(feat_channels[i + 1])
                )

        
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.final_fc = nn.Linear(feat_channels[-1], 3 * self.K)
        
        if self.class_num != 0:
            self.classifer = nn.Linear(feat_channels[-1], self.class_num * self.K)

    def forward(self, f):
        """
        f : (b , feat_channels[0])
        """
        shape = f.shape
        x = f
        for i in range(self.layer_num):
            x = self.fc_layers[i](x)
            if self.norm:
                x = self.norms[i](x)
            x = self.leaky_relu(x)
        
        pc_out = self.final_fc(x)
        pc_out = pc_out.view(shape[0],self.K,-1)
        if self.class_num != 0:
            cls_out = self.classifer(x)
            cls_out = cls_out.view(shape[0],self.K,-1)
            cls_out = F.softmax(cls_out,dim = 2)
            ret = torch.cat([pc_out, cls_out],dim = 2)
            return ret
        return pc_out


class Self_Attention_Layer(nn.Module):
    def __init__(self,in_feature,M=64):
        super(Self_Attention_Layer, self).__init__()
        self.MLP_Q = nn.Linear(in_feature,M)
        self.MLP_K = nn.Linear(in_feature,M)
        self.MLP_V = nn.Linear(in_feature,in_feature)
        self.M = M
        self.in_feature = in_feature
    
    def forward(self,features):
        """
        features: (b , K , in_feature) K -> 32
        """
        Q = self.MLP_Q(features)  # (b , K , M)
        K = self.MLP_K(features)  # (b , K , M)
        V = self.MLP_V(features)  # (b , K , in_feature)

        Q = Q.permute(0,2,1) # (b , M , K)
        score = K.matmul(Q)
        score = score / (self.M ** 0.5)
        
        self.attention = F.softmax(score,dim=2) # (b , K , K)

        new_feature = self.attention.matmul(V)
        new_feature = features + new_feature
        return new_feature


class PC_MLP_4_Gen_Subcloud_add_method_with_modified_attention(nn.Module):
    def __init__(self,feat_channels, K, Z_dim, class_num,attention_module = Self_Attention_Layer):
        super(PC_MLP_4_Gen_Subcloud_add_method_with_modified_attention, self).__init__()
        self.attention = attention_module(feat_channels[0])
        self.linear_pc = nn.Linear(3,feat_channels[0])
        self.linear_class = nn.Linear(class_num,feat_channels[0])
        self.linear_z = nn.Linear(Z_dim,feat_channels[0])
        self.MLP = nn.ModuleList([])
        self.layer_num = len(feat_channels) - 1
        self.K = K
        for i in range(self.layer_num):
            self.MLP.append(
                nn.Linear(feat_channels[i],feat_channels[i + 1])
            )
        
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.final_fc = nn.Linear(feat_channels[-1], 3 * self.K)
    
    def forward(self,pc_0,z):
        pc = pc_0[:,:,:3]
        classes = pc_0[:,:,3:]
        f_0 = self.linear_pc(pc)
        f_1 = self.linear_z(z)
        f_2 = self.linear_class(classes)
        x = f_0 + f_1 + f_2
        x = self.leaky_relu(x)
        x = self.attention(x)
        shape = x.shape
        x = x.view(shape[0] * shape[1], -1)
        for i in range(self.layer_num):
            x = self.MLP[i](x)
            x = self.leaky_relu(x)
        
        x = self.final_fc(x)
        x = x.view(-1,self.K, 3)
        pc = pc.view(shape[0] * shape[1], 1, 3)
        pc = pc.expand(shape[0] * shape[1], self.K, 3)
        pc = torch.add(pc,x)
        pc = pc.view(shape[0],-1,3)
        return pc

