from pathlib import Path
import numpy as np
import torch

def experience(self):
    model = self.model
    z,f_pc_0,f_pc_1 = model.inference()
    z_path = Path("./z.npy")
    pc_0_path = Path("./pc_0.pts")
    pc_1_path = Path("./pc_1.pts")
    with z_path.open("wb") as f:
        torch.save(z[0],f)
    with pc_0_path.open("w") as f:
        np.savetxt(f,f_pc_0[0].cpu().numpy())
    with pc_1_path.open("w") as f:
        np.savetxt(f,f_pc_1[0].cpu().numpy())
        