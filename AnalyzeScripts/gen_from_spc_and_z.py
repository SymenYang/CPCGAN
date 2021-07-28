from pathlib import Path
import numpy as np
import torch

def modify_pc_0(pc_0):
    return pc_0

def experience(self):
    model = self.model
    # load z and structure point cloud
    z_path = Path("./z.npy")
    pc_0_path = Path("./pc_0.pts")
    z,pc_0 = None,None
    with z_path.open("rb") as f:
        z = torch.load(f).view(1,-1)
    with pc_0_path.open("r") as f:
        pc_0 = np.loadtxt(f)
    
    # Modify the structure point cloud
    pc_0 = modify_pc_0(pc_0)
    # To GPU
    z = z.cuda()
    pc_0 = torch.from_numpy(pc_0).float().view(1,-1,7).cuda()
    
    f_pc_1 = model.gen_from_given_z_and_pc0(z,pc_0)

    # Save generated point cloud
    pc_1_path = Path("./pc_1.pts")
    with pc_1_path.open("w") as f:
        np.savetxt(f,f_pc_1[0].cpu().numpy())
    