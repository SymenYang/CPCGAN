from DLNest.Common.DatasetBase import DatasetBase

import torch.utils.data as data
import torch
import numpy as np
import json
from pathlib import Path



class CPCGAN_Dataset(data.Dataset):
    def __init__(self,args : dict):
        self.args = args["dataset_config"]
        root = (Path.cwd() / "../..").resolve() # The root of the relative path is the project path

        self.cat = {}
        with (root / self.args["category_information_path"]).open('r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if self.args["class_choice"] != []:
            self.cat = {key:value for key,value in self.cat.items() if key in self.args["class_choice"]}
        
        self.data_path = []
        for item in self.cat:
            dir_point = root / self.args["full_pointcloud_path"] / self.cat[item] 
            dir_seg = root / self.args["semantic_label_path"] / self.cat[item]
            dir_sampled = root / self.args["pointcloud_sample_path"] / self.cat[item]
            dir_sem_sampling = root / self.args["semantic_avg_sampled_pointcloud_path"] / self.cat[item]
            fns = dir_point.iterdir()

            for fn in fns:
                token = fn.stem
                self.data_path.append((
                    dir_point / (token + ".pts"),
                    dir_seg / (token + ".seg"),
                    dir_sampled / (token + ".sam"),
                    dir_sem_sampling / (token + ".pts")
                ))
        
        meta_path = root / self.args["semantic_avg_sampled_pointcloud_path"] / "meta.json"
        self.meta = json.load(meta_path.open('r'))
        self.max_sem_classes = 0
        for item in self.meta:
            if not item in self.cat:
                continue
            self.max_sem_classes = max(self.max_sem_classes,self.meta[item]["max_sem_classes"])
        self.ones = torch.sparse.torch.eye(self.max_sem_classes)

    def __getitem__(self,index : int):
        fn = self.data_path[index]
        full_pointcloud = np.loadtxt(fn[0]).astype(np.float32)
        sampled_id = np.loadtxt(fn[2]).astype(np.int64)
        full_pointcloud = full_pointcloud[sampled_id]

        sem_pointdatas = np.loadtxt(fn[3]).astype(np.float32)
        sem_pointclouds = sem_pointdatas[:,:3]
        sem_pointsem = torch.from_numpy(sem_pointdatas[:,3:].reshape(len(sem_pointdatas)).astype(np.int64))
        sem_labels = self.ones.index_select(0,sem_pointsem - 1)
        sem_pointclouds = np.concatenate([sem_pointclouds,sem_labels],axis=1)
        return full_pointcloud,sem_pointclouds
    
    def __len__(self):
        return len(self.data_path)

class CPCGAN_Datas(DatasetBase):
    def init(self,args : dict):
        """
        init dataset
        input: 
            args: dict of task argumentations
        
        output:
            dict,Dataloader,Dataloader
            dict: infomation which dataset want to tell model when model init
            Dataloaders: Dataloaders for training,validation. If no validation, return two same dataloaders
        """
        self.train_set = CPCGAN_Dataset(args)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size = args["batch_size"],
            shuffle = True,
            pin_memory = True,
            num_workers = 4,
            drop_last = True
        )
        print("Finish dataset initialize")
        return {"class_num" : self.train_set.max_sem_classes},self.train_loader,self.train_loader