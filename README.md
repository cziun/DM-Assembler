# DM-Assembler: Leveraging Domain Motif Assembler for Multi-objective, Multi-domain and Explainable Molecular Design

![framework](figure/framework.png)


# Installation

## 1. Molecule Generation

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

## 2. Property Simulation

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


# Model Training

> **Note:** These codes run in the environment `gen`.

## 1. Vocab Construction

You can use the following command to construct the motif vocabulary:

```
python vocab_generation.py --data '/path/to/your/dataset' --vocab_size size_of_the_vocabulary --vocab_path '/path/to/vocab'
```

where `vocab_size` can be changed according to your need. After running, you will get a text file of vocabulary in which the lines record the mined motif. Each row consists of three items: the SMILES of the motif, the number of atoms in the motif, and the frequency of the motif in the dataset. 

## 2. Data Preprocess

You can use the following command to process the atom-level data into motif-level data using the vocabulary we built:

```
python preprocess.py --data '/path/to/your/dataset' --vocab_path '/path/to/vocab' --arr_x_path '/path/to/x' --arr_adj_path '/path/to/adj' --valid_idx_path '/path/to/valid_idx' --test_nx_path '/path/to/test_nx'
```

where `arr_x_path` and `arr_adj_path` represent the path of motif compositions and connections, respectively, and `test_nx_path` is the path of the test fragment-level graphs. 

















