from DLNest.Common.ModelBaseTorch import ModelBaseTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Subnets import PC_MLP,PC_MLP_4_Gen_Subcloud_add_method_with_modified_attention
from GAN_network import Discriminator,Discriminator_avg_pooling,Generator
from Gradient_penalty import GradientPenalty
from FPD.FPD import calculate_fpd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class CPCGAN(ModelBaseTorch):
    def init(self,args : dict,datasetInfo : dict = None):
        self.pre_epoch = 500
        self.args = args["model_config"]
        self.epochs = args["epochs"]
        self.batch_size = args["batch_size"]
        self.class_num = datasetInfo["class_num"]
        self.device = 0
        
        self.model_init()
        self.visualize_init(args)
        print("Finish model initialize")
    
    def initOptimizer(self):
        self.optimizerD_0 = optim.Adam(
            self.D_0.parameters(),
            lr = self.args['lr'],
            betas = (0,0.99)
        )
        self.optimizerD_1 = optim.Adam(
            self.D_1.parameters(),
            lr = self.args['lr'],
            betas = (0,0.99)
        )
        self.optimizerG = optim.Adam(
            self.G.parameters(),
            lr = self.args['lr'],
            betas = (0,0.99)
        )

    def model_init(self):
        self.G_FEAT_0 = self.args["G_FEAT_0"]
        self.Z_DIM = self.G_FEAT_0[0]
        self.G_FEAT_1 = self.args["G_FEAT_1"]
        self.D_FEAT_0 = self.args["D_FEAT_0"]
        self.D_FEAT_1 = self.args["D_FEAT_1"]
        self.D_FEAT_0[0] += self.class_num
        assert self.D_FEAT_0[0] == self.class_num + 3

        self.Net_0 = PC_MLP(
            self.G_FEAT_0,self.args['K'],
            self.class_num
        )
        self.Net_1 = PC_MLP_4_Gen_Subcloud_add_method_with_modified_attention(
            self.G_FEAT_1,
            int(2048 / self.args['K']),
            self.Z_DIM,
            self.class_num
        )

        self.D_0 = Discriminator_avg_pooling(self.D_FEAT_0,self.class_num)
        self.D_0 = self.register(self.D_0)
        self.D_1 = Discriminator(self.D_FEAT_1)
        self.D_1 = self.register(self.D_1)

        self.G = Generator(self.Net_0,self.Net_1,self.args['K'])
        self.G = self.register(self.G)

        self.GP = GradientPenalty(self.args['lambdaGP'],device=self.device)

    def visualize_init(self,args):
        colors = np.array([(51,204,96),(115,153,126),(145,255,89),(228,153,255),(122,51,204),(144,255,217),(47,204,104),(0,154,209)])
        self.colors = colors[:self.class_num]
        self.writer = SummaryWriter(Path(__file__).parent.parent)

    def initLog(self):
        self.log = {
            "D_loss_0" : [],
            "D_loss_1" : [],
            "D_loss" : [],
            "G_loss_0" : [],
            "G_loss_1" : [],
            "G_loss" : [],
            "FPD" : []
        }
        return self.log

    def pre_train(self,data,log):
        pc_1,pc_0 = data

        for d_iter in range(self.args["D_iter"]):
            self.D_0.zero_grad()
            self.D_1.zero_grad()
            z = torch.randn(self.batch_size,self.Z_DIM).cuda()

            shape = z.shape
            expanded_z = z.view(shape[0],1,shape[1])
            expanded_z = expanded_z.expand(shape[0],self.args["K"],shape[1])

            with torch.no_grad():
                f_pc_0 = self.Net_0(z)
                f_pc_1 = self.Net_1(pc_0,expanded_z)
            
            D_real_0 = self.D_0(pc_0)
            D_real_0_mean = D_real_0.mean()

            D_real_1 = self.D_1(pc_1)
            D_real_1_mean = D_real_1.mean()

            D_fake_0 = self.D_0(f_pc_0)
            D_fake_0_mean = D_fake_0.mean()

            D_fake_1 = self.D_1(f_pc_1)
            D_fake_1_mean = D_fake_1.mean()

            gp_loss_0 = self.GP(
                self.D_0,
                pc_0.data,
                f_pc_0.data
            )

            gp_loss_1 = self.GP(
                self.D_1,
                pc_1.data,
                f_pc_1.data
            )

            d_loss_0 = - D_real_0_mean + D_fake_0_mean + gp_loss_0
            d_loss_1 = - D_real_1_mean + D_fake_1_mean + gp_loss_1
            d_loss_0.backward()
            d_loss_1.backward()
            self.optimizerD_0.step()
            self.optimizerD_1.step()
        log["D_loss_0"].append(d_loss_0.item()) 
        log["D_loss_1"].append(d_loss_1.item()) 
        log["D_loss"].append(0.0)
        
        self.Net_0.zero_grad()
        self.Net_1.zero_grad()
        z = torch.randn(self.batch_size,self.Z_DIM).cuda()
        f_pc_0 = self.Net_0(z)
        f_pc_1 = self.Net_1(pc_0,expanded_z)
        G_fake_0 = self.D_0(f_pc_0)
        G_fake_0_mean = G_fake_0.mean()

        G_fake_1 = self.D_1(f_pc_1)
        G_fake_1_mean = G_fake_1.mean()

        g_loss_0 = -G_fake_0_mean
        g_loss_1 = -G_fake_1_mean
        g_loss_0.backward()
        g_loss_1.backward()

        self.optimizerG.step()
        self.G.pc_0 = f_pc_0
        self.G.pc_1 = f_pc_1
        
        log["G_loss_0"].append(g_loss_0.item()) 
        log["G_loss_1"].append(g_loss_1.item()) 
        log["G_loss"].append(0.0)

    def runOneStep(self,data,log : dict,iter : int,epoch : int):
        self.now_epoch = epoch
        if epoch < self.pre_epoch:
            self.pre_train(data,log)
            return
        
        pc_1,pc_0 = data

        for d_iter in range(self.args["D_iter"]):

            self.D_0.zero_grad()
            self.D_1.zero_grad()

            z = torch.randn(self.batch_size,self.Z_DIM).cuda()
            with torch.no_grad():
                f_pc_0,f_pc_1 = self.G(z)
            
            D_real_0 = self.D_0(pc_0)
            D_real_0_mean = D_real_0.mean()

            D_real_1 = self.D_1(pc_1)
            D_real_1_mean = D_real_1.mean()

            D_fake_0 = self.D_0(f_pc_0)
            D_fake_0_mean = D_fake_0.mean()

            D_fake_1 = self.D_1(f_pc_1)
            D_fake_1_mean = D_fake_1.mean()

            gp_loss_0 = self.GP(
                self.D_0,
                pc_0.data,
                f_pc_0.data
            )

            gp_loss_1 = self.GP(
                self.D_1,
                pc_1.data,
                f_pc_1.data
            )

            d_loss_0 = - D_real_0_mean + D_fake_0_mean + gp_loss_0
            d_loss_1 = - D_real_1_mean + D_fake_1_mean + gp_loss_1
            d_loss = d_loss_0 * self.args["lambdaD"] + d_loss_1
            d_loss.backward()
            self.optimizerD_0.step()
            self.optimizerD_1.step()
        log["D_loss_0"].append(d_loss_0.item()) 
        log["D_loss_1"].append(d_loss_1.item()) 
        log["D_loss"].append(d_loss.item())

        self.G.zero_grad()
        z = torch.randn(self.batch_size,self.Z_DIM).cuda()
        f_pc_0,f_pc_1 = self.G(z)

        G_fake_0 = self.D_0(f_pc_0)
        G_fake_0_mean = G_fake_0.mean()

        G_fake_1 = self.D_1(f_pc_1)
        G_fake_1_mean = G_fake_1.mean()

        g_loss_0 = -G_fake_0_mean
        g_loss_1 = -G_fake_1_mean
        g_loss = g_loss_0 * self.args["lambdaG"] + g_loss_1
        g_loss.backward()
        self.optimizerG.step()
        
        log["G_loss_0"].append(g_loss_0.item()) 
        log["G_loss_1"].append(g_loss_1.item()) 
        log["G_loss"].append(g_loss.item())

    def inference(self,sizes = 1):
        z = torch.randn(sizes,self.Z_DIM).cuda()
        f_pc_0,f_pc_1 = None,None
        with torch.no_grad():
            f_pc_0,f_pc_1 = self.G(z)
        return z,f_pc_0,f_pc_1

    def gen_from_given_z_and_pc0(self,z,pc_0):
        f_pc_1 = None
        shape = z.shape
        with torch.no_grad():
            z2 = z.view(shape[0],1,shape[1])
            z2 = z2.expand(shape[0],self.G.K,shape[1])
            f_pc_1 = self.G.Net_1(pc_0,z2)
        return f_pc_1

    def visualize(self,log : dict, iter : int, epoch : int):
        self.writer.add_scalar("D_loss_0",log["D_loss_0"][-1],len(log["D_loss_0"]) - 1)
        self.writer.add_scalar("D_loss_1",log["D_loss_1"][-1],len(log["D_loss_1"]) - 1)
        self.writer.add_scalar("D_loss",log["D_loss"][-1],len(log["D_loss"]) - 1)
        self.writer.add_scalar("G_loss_0",log["G_loss_0"][-1],len(log["G_loss_0"]) - 1)
        self.writer.add_scalar("G_loss_1",log["G_loss_1"][-1],len(log["G_loss_1"]) - 1)
        self.writer.add_scalar("G_loss",log["G_loss"][-1],len(log["G_loss"]) - 1)

    def run_FPD(self):
        fake_pointclouds = torch.Tensor([])
        max_iter = int(4096 / self.batch_size)
        for i in range(max_iter):
            z = torch.randn(self.batch_size,self.Z_DIM).cuda()
            p1 = None
            with torch.no_grad():
                _,p1 = self.G(z)
                p1 = p1.cpu()
            fake_pointclouds = torch.cat((fake_pointclouds, p1), dim=0)

        fpd = calculate_fpd(fake_pointclouds, PointNet_pretrained_path = self.args["PointNet_weights"],statistic_save_path=self.args["FPD_path"], device=self.device)
        return fpd

    def validate(self,valLoader,log : dict):
        if self.now_epoch < self.pre_epoch:
            fpd = 10000.0
        else:
            fpd = self.run_FPD()
        log["FPD"].append(fpd)
        self.writer.add_scalar("FPD",log["FPD"][-1],len(log["FPD"]) - 1)
