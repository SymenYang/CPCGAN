import pathlib
from pathlib import Path
import json
import random

root = Path(__file__).absolute().parent.parent
config_file = root / "dataset_config.json"

def sample_k_points(config,k=2048):
    cat = {}

    cat_path = root / config["category_information_path"]
    with cat_path.open("r") as f:
       for line in f:
           ls = line.strip().split()
           cat[ls[0]] = ls[1]    
    
    meta = {}
    for item in cat:
        meta[item] = []
        dir_point = root / Path(config["full_pointcloud_path"] ) / cat[item] 
        dir_seg = root / Path(config["semantic_label_path"] ) / cat[item]
        dir_sampling = root / Path(config["pointcloud_sample_path"] ) / cat[item]

        fns = dir_point.iterdir()

        for fn in fns:
            token = fn.stem
            meta[item].append((
                dir_point / (token + '.pts'),
                dir_seg / (token + '.seg'),
                dir_sampling / (token + '.sam'),
                dir_sampling
                ))
    
    for cls_key in meta:
        print(cls_key)
        cls_list = meta[cls_key]
        for item in cls_list:
            ifp = item[0].open('r')
            if not item[3].exists():
                item[3].mkdir(parents=True,exist_ok=True)
            ofp = item[2].open('w')

            lines = ifp.readlines()
            lst = [str(i) + '\n' for i in range(len(lines))]
            if len(lst) < k:
                ifp.close()
                ofp.close()
                item[0].unlink()
                item[1].unlink()
                item[2].unlink()
                print("points not enough ",item[0])
                continue
            slines = random.sample(lst,k)
            ofp.writelines(slines)
            ifp.close()
            ofp.close()

if __name__ == "__main__":
    with config_file.open("r") as f:
        config = json.load(f)["dataset_config"]
        print("Sampling ground truth points")
        sample_k_points(config)

# sudo docker run -it --gpus=all -p 10222:22 -p 10223:8097 -v /home/imc/symenyang/Projects/Phoneix:/root/code -v /home/imc/symenyang/Datas:/root/data --name CPCGAN_release_test common:latest /bin/bash