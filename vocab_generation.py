import os
from copy import copy
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import pandas as pd
from rdkit import Chem

from utils.mol_utils import smiles2mol, mol2smiles, get_submol, MAX_VALENCE, cnt_atom, GeneralVocab


def load_molecule_props(csv_path):
    molecule_scores = {}
    df = pd.read_csv(csv_path)
    for smi, score in zip(df.iloc[:, 0], df.iloc[:, 2]):  # Normalized property value
        mol = smiles2mol(smi, sanitize = True)
        smi1 = mol2smiles(mol)
        molecule_scores[smi1] = score
    return molecule_scores


class MolInPiece:
    def __init__(self, mol): 
        self.mol = mol
        self.smi = mol2smiles(mol)
        self.pieces, self.pieces_smis = {}, {}  # pid is the key (init by all atom idx)
        for atom in mol.GetAtoms():
            idx, symbol = atom.GetIdx(), atom.GetSymbol()  # idx from 0 to len
            self.pieces[idx] = { idx: symbol }
            self.pieces_smis[idx] = symbol
        self.inversed_index = {} # assign atom idx to pid
        for aid in range(mol.GetNumAtoms()):
            for key in self.pieces:
                piece = self.pieces[key]
                if aid in piece:
                    self.inversed_index[aid] = key
        self.dirty = True
        self.smi2pids = {} # not public, record neighboring graphs and their pids

    def get_nei_pieces(self):
        nei_pieces, merge_pids = [], []
        for key in self.pieces:
            piece = self.pieces[key]
            local_nei_pid = []
            for aid in piece:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx in piece or nei_idx > aid:   # only consider connecting to former atoms
                        continue
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_piece = copy(piece)
                new_piece.update(self.pieces[nei_pid])
                nei_pieces.append(new_piece)
                merge_pids.append((key, nei_pid))
        return nei_pieces, merge_pids
    
    def get_nei_smis(self):
        if self.dirty:
            nei_pieces, merge_pids = self.get_nei_pieces()
            nei_smis, self.smi2pids = [], {}
            for i, piece in enumerate(nei_pieces):
                submol = get_submol(self.mol, piece)
                smi = mol2smiles(submol)
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis
    
    def merge(self, smi):
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids:
                if pid1 in self.pieces and pid2 in self.pieces: # possibly del by former
                    self.pieces[pid1].update(self.pieces[pid2])
                    self.pieces_smis[pid1] = smi
                    for aid in self.pieces[pid2]:
                        self.inversed_index[aid] = pid1
                    del self.pieces[pid2]
                    del self.pieces_smis[pid2]
        self.dirty = True   # revised

    def get_smis_pieces(self):
        # return list of tuple(smi, idxs)
        res = []
        for pid in self.pieces_smis:
            smi = self.pieces_smis[pid]
            group_dict = self.pieces[pid]
            idxs = list(group_dict.keys())
            res.append((smi, idxs))
        return res


def property_freq_score(mol):
    freqs = {}
    prop_scores = {}
    nei_smis = mol.get_nei_smis()
    mol_smi = mol.smi
    for smi in nei_smis:
        freqs.setdefault(smi, 0)
        freqs[smi] += 1
        prop_scores.setdefault(smi, 0)
        prop_scores[smi] = molecule_scores[mol_smi]
    return freqs, prop_scores, mol


def graph_bpe_property_aware(fname, vocab_len, vocab_path, alpha, cpus):

    print('Weight of Frequency: ', alpha)
    print('Weight of Property: ', 1 - alpha)

    # load molecules
    print(f'Loading mols from {fname} ...')
    df = pd.read_csv(fname)
    if 'smiles' not in df.columns:
        raise ValueError("Missing smiles column in CSV file")
    smis = df['smiles'].astype(str).tolist()

    # init to atoms
    mols = [MolInPiece(smiles2mol(smi, sanitize = True)) for smi in smis if '+' not in smi]
    # loop
    selected_smis, details = list(MAX_VALENCE.keys()), {}   # details: <smi: [atom cnt, score] 
    for atom in selected_smis:
        details[atom] = [1, 0]  # score of single atom is not calculated
    # property-aware bpe process
    add_len = vocab_len - len(selected_smis)
    pool = mp.Pool(cpus)
    for i in tqdm(range(add_len)):
        res_list = pool.map(property_freq_score, mols)  # each element is (freq, prop, mol)
        score_sums, freqs, prop_scores, mols = {}, {}, {}, []
        for freq, prop_score, mol in res_list:
            mols.append(mol)
            for smi, score in prop_score.items():
                prop_scores.setdefault(smi, 0)
                prop_scores[smi] += score
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
                
        for smi in freqs:
            score_sums.setdefault(smi, 0)
            freq = freqs[smi]
            prop = prop_scores[smi]
            sc = alpha * freq + (1 - alpha) * prop
            score_sums[smi] = sc

        # find the piece to merge
        max_score, merge_smi = 0, ''
        for smi in score_sums:
            score = score_sums[smi]
            if score > max_score:
                max_score = score
                merge_smi = smi

        # merge
        for mol in mols:
            mol.merge(merge_smi)
        selected_smis.append(merge_smi)
        details[merge_smi] = [cnt_atom(merge_smi), max_score]
    print('sorting vocab by atom num')
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)
    pool.close()
    dir_path = os.path.dirname(vocab_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(vocab_path, 'w') as fout:
        fout.writelines(list(map(lambda smi: f'{smi}\t{details[smi][0]}\t{details[smi][1]}\n', selected_smis)))
    print('Vocab saved in: ', vocab_path)
    return selected_smis, details


class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        self.vocab_dict = {}
        self.idx2piece, self.piece2idx = [], {}
        for line in lines:
            smi, aton_num, score = line.strip().split('\t')
            self.vocab_dict[smi] = (int(aton_num), float(score))
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)
        self.pad, self.end = '<pad>', '<s>'
        for smi in [self.pad, self.end]:
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)
        # for fine-grained level (atom level)
        self.atom_pad = '<pad>'
        self.chem_vocab = GeneralVocab(atom_special=[self.atom_pad])
        # self.chem_vocab = GeneralVocab()

    def tokenize(self, mol, return_idx=False):
        if isinstance(mol, str):
            mol = smiles2mol(mol, sanitize = True)
        rdkit_mol = mol
        mol = MolInPiece(mol)
        while True:
            nei_smis = mol.get_nei_smis()
            max_score, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                sc = self.vocab_dict[smi][1]
                if sc > max_score:
                    max_score, merge_smi = sc, smi
            if max_score == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_pieces()
        # construct reversed index
        aid2pid = {}
        for pid, piece in enumerate(res):
            _, aids = piece
            for aid in aids:
                aid2pid[aid] = pid
        # construct adjacent matrix
        ad_mat = [[0 for _ in res] for _ in res]
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                nei_id = nei.GetIdx()
                i, j = aid2pid[aid], aid2pid[nei_id]
                if i != j:
                    ad_mat[i][j] = ad_mat[j][i] = 1

        np.random.shuffle(res)
        res.insert(0, (self.end, []))
        res.append((self.end, []))

        if not return_idx:
            return res
        piece_idxs = [self.piece_to_idx(x[0]) for x in res] 
        group_idxs = [x[1] for x in res]
        return piece_idxs, group_idxs

    def idx_to_piece(self, idx):
        return self.idx2piece[idx]
        
    def piece_to_idx(self, piece):
        return self.piece2idx[piece]
    
    def pad_idx(self):
        return self.piece2idx[self.pad]

    def end_idx(self):
        return self.piece2idx[self.end]

    def atom_pad_idx(self):
        return self.chem_vocab.atom_to_idx(self.atom_pad)
    
    def num_piece_type(self):
        return len(self.idx2piece)

    def num_atom_type(self):
        return self.chem_vocab.num_atom_type()
    
    def __call__(self, mol, return_idx=False):
        return self.tokenize(mol, return_idx)

    def __len__(self):
        return len(self.idx2piece)


def parse():
    """parse command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--vocab_size', type=int, help='Length of vocab')
    parser.add_argument('--vocab_path', type=str, help='Path to save vocab')
    parser.add_argument('--weight', type=float, default=1, help='Weight for frequency (1-weight for property)')
    parser.add_argument('--workers', type=int, default=16, help='Number of cpus to use')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    molecule_scores = load_molecule_props(args.data)
    graph_bpe_property_aware(args.data, args.vocab_size, args.vocab_path, args.weight, args.workers)
