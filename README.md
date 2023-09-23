Source code of our NeurIPS 2023 Spotlight paper "Provable Training for Graph Contrastive Learning"

# Environment Settings

Here we list some important python packages we used:
```
torch == 1.12.1
# PyG
torch-geometric == 2.2.0
torch-cluster == 1.6.0
torch-scatter == 2.1.0
torch-sparse == 0.6.16
torch-spline-conv == 1.2.1
#DGL
dgl == 1.0.2
PyGCL == 0.1.2
```
To install PyG, we suggest to follow the official guide: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
To install DGL, please follow https://docs.dgl.ai/install/index.html
# Usage
Fisrt, make the directories for datasets and bounds to save
``` bash
mkdir ~/datasets
mkdir ~/datasets/bounds
```
Then, go into the directory of a model. If you want to set the parameters, you should modify the ocnfiguration files in the directory ("config.yaml" for GRACE, "config/default.yaml" for COSTA, "param/{dataset_name}.json" for GCA and ProGCL). The following is the command line to run each model (dataset used is Cora for example):
```bash
# GRACE
cd GRACE
# original GRACE
python train.py --dataset Cora --gpu_id 0
# GRACE + POT
python train.py --dataset Cora --gpu_id 0 --use_pot --kappa 0.4

# GCA/ProGCL
cd GCA # cd ProGCL
# original GCA
python train.py --dataset Cora --param local:cora.json --device cuda:0
# GCA + POT
python train.py --dataset Cora --param local:cora.json --device cuda:0 --use_pot --kappa 0.3

# COSTA
cd COSTA/src
# original COSTA
python main.py --dataset Cora --gpu_id 0
# COSTA + POT
python main.py --dataset Cora --gpu_id 0 --use_pot --kappa 0.2
```
The result will be appended to the file "res/{dataset_name}_base_temp.csv" and "res/{dataset_name}_pot_temp.csv" respectively. You can also set the parameter "save_file" to specify the file to save results. We use minibatch to reduce the memory occupation, you can modify it in the code. To use minibatch for POT, set "pot_batch", usually 256/512/1024 will work:
```bash
# GCA when use pot_batch on BlogCatalog
python train.py --dataset BlogCatalog --gpu_id 0 --use_pot --kappa 0.3 --pot_batch 1024
```