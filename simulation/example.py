import argparse

from tartarus import pce
from tartarus import tadf
from tartarus import docking
from tartarus import reactivity


def evaluate(args):

    dataset = args.dataset
    smiles = args.smiles
    print('dataset: ', dataset)
    print('smiles: ', smiles)

    if dataset == 'hce':
        pce_1, pce_2, sas, pce_pcbm_sas, pce_pcdtbt_sas = pce.get_properties(smiles)
        print(f"pce_1: {pce_1}, pce_2: {pce_2}, sas: {sas}, pce_pcbm_sas: {pce_pcbm_sas}, pce_pcdtbt_sas: {pce_pcdtbt_sas}")
    elif dataset == 'gdb13':
        st, osc, combined = tadf.get_properties(smiles)
        print(f"singlet-triplet value: {st}, oscillator strength: {osc}, multi-objective value: {combined}")
    elif dataset == 'snb60k':
        Ea, Er, sum_Ea_Er, diff_Ea_Er, sa_score  = reactivity.get_properties(smiles, n_procs=50)  # set number of processes
        print(f"Ea: {Ea}, Er: {Er}, sum_Ea_Er: {sum_Ea_Er}, diff_Ea_Er: {diff_Ea_Er}, sa_score: {sa_score}")
    elif dataset == 'dtp':
        score_qvina = docking.perform_calc_single(smiles, '4lde', docking_program='qvina')
        score_smina = docking.perform_calc_single(smiles, '4lde', docking_program='smina')
        print(f"4lde_score_qvina: {score_qvina}, 4lde_score_smina: {score_smina}")
    else:
        print('Wrong dataset input')


def parse():
    """parse command"""
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataset', type=str, default="dtp")
    parser.add_argument('--smiles', type=str, default="OC1=C2N=COC2=CC2=C1OC=C2")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    evaluate(args)