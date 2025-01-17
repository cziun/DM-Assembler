<h2 align="center"> DM-Assembler: Leveraging Domain Motif Assembler for Multi-objective, Multi-domain and Explainable Molecular Design</a></h2>
<h5 align="center">

## Overview

![framework](figure/framework.png)


## Installation

### 1. Molecule Generation

First, create a virtual environment and activate the environment:

```
conda create -n gen python=3.7
conda activate gen
```

Then, install the packages:

```
pip install -r requirements.txt
```

Finally, install pytorch_geometric:

```
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric
```

### 2. Property Simulation

First, create a virtual environment using the provided environment configuration and activate the environment:

```
cd simulation
conda env create -f environment.yml
conda activate tartarus
```

Second, set environment variables:

```
export XTBHOME=$CONDA_PREFIX
source $CONDA_PREFIX/share/xtb/config_env.bash
```

Optionally, you can configure the environment variables to be set automatically when you activate this environment:

```
echo "export XTBHOME=$CONDA_PREFIX" > $CONDA_PREFIX/etc/conda/activate.d/env.sh
echo "source $CONDA_PREFIX/share/xtb/config_env.bash" >> $CONDA_PREFIX/etc/conda/activate.d/env.sh
```

Finally, ensure that docking task executables have the correct permissions:

```
chmod 777 tartarus/data/qvina
chmod 777 tartarus/data/smina
```


## Model Training

> **Note:** The codes for model training and molecule sampling run in the environment `gen`.

### 1. Vocab Construction

You can use the following command to construct the motif vocabulary:

```
python vocab_generation.py --data '/path/to/your/dataset' --vocab_size size_of_the_vocabulary --vocab_path '/path/to/vocab'
```

where `vocab_size` can be changed according to your need. After running, you will get a text file of vocabulary in which the lines record the mined motif. Each row consists of three items: the SMILES of the motif, the number of atoms in the motif, and the frequency of the motif in the dataset. 

### 2. Data Preprocess

You can use the following command to process the atom-level data into motif-level data using the vocabulary we built:

```
python preprocess.py --data '/path/to/your/dataset' --vocab_path '/path/to/vocab' --arr_x_path '/path/to/x' --arr_adj_path '/path/to/adj'
```

where `arr_x_path` and `arr_adj_path` represent the path of motif compositions and connections, respectively. 

### 3. Coarse-grained Training

The configuration is provided in the `config/` directory in `yaml` format. To train the coarse-grained score-based model for motif-connection generation, first modify `config/${dataset}.yaml`. Then, you can run the following command:

```
python main.py --type train --config ${train_config}
```

After running, you will get the `.ckpt` for a model, which is used for motif-connection generation.

### 4. Fine-grained Training

To train the fine-grained model for the assembly of complete molecules, you can run the following command:

```
python trainer_bond_recovery.py --config sample_${dataset} --vocab_path '/path/to/vocab' --train_set '/path/to/train' --valid_set '/path/to/valid' --test_set '/path/to/test'
```

After running, you will get the corresponding `.ckpt` file.

## Molecule Sampling

To generate complete molecules using the above trained multi-granularity model, run the following command:

```
python main.py --type sample --config sample_${dataset} --ckpt_train_path '/path/to/coarse/ckpt' --ckpt_bond_path '/path/to/fine/ckpt' --vocab '/path/to/vocab' --output '/path/to/mol/output'
```

After running, you will get a text file of generated molecules in `output`.


## Property Simulation

> **Note:** The codes for molecular property simulation run in the environment `tartarus`.

You can use the following command to get the properties of a molecule:

```
cd simulation
python example.py --dataset 'dtp' --smiles 'OC1=C2N=COC2=CC2=C1OC=C2'
```

where the candidate of dataset are 'hce', 'gdb13', 'snb60k', and 'dtp'.

## Checkpoints

The checkpoints for my model can be found at the following link: [Checkpoints](https://drive.google.com/drive/folders/1NCvBiymP4eDsNmMcbymNj7oLTjBbD3vt?usp=drive_link). Please download it to continue with the model reproduction or evaluation.


## Acknowledgements

We would like to express our gratitude to the related projects, research and development personnel:

1. https://github.com/aspuru-guzik-group/Tartarus
2. https://github.com/THUNLP-MT/PS-VAE
3. https://github.com/harryjo97/GDSS










