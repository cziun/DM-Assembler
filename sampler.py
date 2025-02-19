import os
import time
import torch
import numpy as np

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_device, load_ckpt, load_seed, load_model_from_ckpt, load_sampling_fn, load_data
from utils.mol_utils import load_smiles, canonicalize_smiles, mol2smiles
from utils.graph_utils import init_flags, quantize_mol
from vocab_generation import Tokenizer
from bond.BondRecovery import BondRecovery


class Sampler_mol(object):
    def __init__(self, config):
        self.config = config
        self.device = load_device()

    def sample(self, args):
        start_time = time.time()

        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(args.ckpt_train_path, self.config, self.device)
        self.configt = self.ckpt_dict['config']

        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device) 
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device)

        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device) 

        tokenizer = Tokenizer(args.vocab)
        checkpoint = torch.load(f'./checkpoints/BondRecovery/{self.config.data.data}/{args.ckpt_bond_path}/best_model.ckpt')
        self.model_bond = BondRecovery(self.config, tokenizer)
        if args.gpus == -1:
            loc = torch.device('cpu')
        else:
            loc = torch.device(f'cuda:{args.gpus}')
        self.model_bond.load_state_dict(checkpoint['state_dict'])
        self.model_bond.to(loc)
        self.model_bond.eval()

        print("Loading models done ......")

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        train_smiles, test_smiles = load_smiles(self.configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles) 

        self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)  # for init_flags
        self.init_flags = init_flags(self.train_graph_list, self.configt, 10000).to(self.device[0])

        x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)

        adj = quantize_mol(adj)

        x = x.cpu()
        x = torch.where(torch.argmax(x, dim=2) == 0, torch.tensor(0), torch.argmax(x, dim=2) + 1)
        x = x.numpy().astype(np.int64)        

        print('Generating molecules ......')

        gen_mols = []
        gen_smiles = []

        if not os.path.exists('output'):
            os.makedirs('output')
        f = open(f'output/{args.output}.txt', 'w')

        for i in range(len(x)):
            motif_ids = x[i] - 1
            adj_inter_motif = adj[i]
            mol = self.gen(motif_ids, adj_inter_motif)
            smi = mol2smiles(mol)
            print(i, ', ', smi)
            gen_mols.append(mol)
            gen_smiles.append(smi)
            f.write(f'{smi}\n')
        f.close()

        end_time = time.time()
        runtime = end_time - start_time
        logger.log(f'Sample time: {runtime}')
        print(f"Running time: {runtime:.2f} seconds")


    def gen(self, motif_ids, adj_inter_motif):
        return self.model_bond.inference_single_z(motif_ids, adj_inter_motif, self.config.generate.add_edge_th, self.device[0])
