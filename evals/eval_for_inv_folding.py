import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='folding', choices=['folding', 'inv_folding'], help='The task type: folding or inverse folding.')

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--results_root', type=str, default='')
parser.add_argument('--grid_eval', action='store_true', help="enable grid evaluation (default: False)")

parser.add_argument('--key_word', type=str, default='')
parser.add_argument('--prefix', type=str, default='struct', help='The prefix of decoded pdb files.')

args = parser.parse_args()

import os

import pandas as pd
import numpy as np

from tqdm import tqdm

import biotite.sequence.io.fasta as fasta

from eval import get_pdb_file, get_pkl_file, cal_bb_rmsd, cal_bb_tm

def get_evaluation_results(gt_dir, results_dir, mode='backbone', filter_x=False, return_plddt=False):
    eval_df = pd.DataFrame(columns=['pdb_name', 'rmsd', 'tm_score'])
    plddt_list = []
    for pdb_file in tqdm(os.listdir(results_dir)):
        if not pdb_file.endswith('.pdb'):
            continue
        
        if 'plddt' in pdb_file:
            pdb_name = '_'.join(pdb_file.split('_')[:-2])
            plddt = float(pdb_file.split('_')[-1][:-4])
            plddt_list.append(plddt)
        else:
            pdb_name = pdb_file[:-4]
        
        gt_pkl_path = os.path.join(gt_dir, f'{pdb_name}.pkl')
        decoded_pdb_path = os.path.join(results_dir, pdb_file)
        
        gt_bb_pos, gt_seq, mask = get_pkl_file(gt_pkl_path, mode=mode)
        resi_mask = mask[:, 1] # use CA atom mask for filtering
        decoded_bb_pos, _, _ = get_pdb_file(decoded_pdb_path, mode=mode)
        
        if filter_x:
            valid_indices = [i for i, aa in enumerate(gt_seq) if aa != 'X']
            gt_bb_pos = gt_bb_pos[valid_indices]
            gt_seq = ''.join([gt_seq[i] for i in valid_indices])
            resi_mask = resi_mask[valid_indices]

        try:
            gt_bb_pos = gt_bb_pos[resi_mask]
            decoded_bb_pos = decoded_bb_pos[resi_mask]
            gt_seq = ''.join([gt_seq[i] for i in range(len(resi_mask)) if resi_mask[i]])
            rmsd = cal_bb_rmsd(gt_bb_pos, decoded_bb_pos)
            tm_score = cal_bb_tm(gt_bb_pos[:, :3, :], decoded_bb_pos[:, :3, :], gt_seq)
        except Exception as e:
            print(gt_seq, resi_mask)
            print(f"Error processing {pdb_name}: {e}")
            rmsd = np.nan
            tm_score = np.nan

        eval_df.loc[len(eval_df)] = {
            'pdb_name': pdb_name,
            'rmsd': rmsd,
            'tm_score': tm_score
        }
    
    if return_plddt:
        eval_df['plddt'] = plddt_list
    return eval_df

def cal_aar(gt_fasta, inv_fold_fasta):

    gt_fasta_file = fasta.FastaFile.read(gt_fasta)
    inv_fold_fasta_file = fasta.FastaFile.read(inv_fold_fasta)

    eval_df = pd.DataFrame(columns=['pdb_name', 'aar'])
    for key in  gt_fasta_file.keys():
        gt_seq = gt_fasta_file[key]
        inverse_fold_seq = inv_fold_fasta_file[key]
        assert len(gt_seq) == len(inverse_fold_seq), f"Ground truth and inverse folded sequences have different lengths: {len(gt_seq)} vs. {len(inverse_fold_seq)}."

        # calculate the amino acid recovery
        match_count = sum(1 for a, b in zip(gt_seq, inverse_fold_seq) if a == b)
        aar = match_count / len(gt_seq)

        eval_df.loc[len(eval_df)] = {
            'pdb_name': key,
            'aar': aar
        }
    
    return eval_df

def main():

    root = args.results_root
    if args.grid_eval:
        grid_list = os.listdir(root)
    else:
        grid_list = ['']
    for grid in grid_list:
        if args.key_word not in grid:
            continue
        results_dir = os.path.join(root, grid)
        eval_file = eval_file = [file for file in os.listdir(results_dir) if file.startswith('eval_results') & file.endswith('csv')]
        if len(eval_file) > 0:
            continue
        print(f'Processing grid: {root}/{grid}')

        if args.task == 'folding':
            
            eval_results = get_evaluation_results(
                gt_dir=f'{args.data_dir}/preprocessed',
                results_dir=f'{results_dir}/decoded_pdb',
                mode='backbone',
                filter_x=True if 'hdprot' in results_dir else False
            )

            mean_rmsd = eval_results['rmsd'].mean()
            mean_tm = eval_results['tm_score'].mean()
            print(f'Mean RMSD: {mean_rmsd:.3f}, Mean TM-score: {mean_tm:.3f}')
            eval_results.to_csv(os.path.join(results_dir, f'eval_results-rmsd={mean_rmsd:.3f}-tm={mean_tm:.3f}.csv'), index=False)
        
        elif args.task == 'inv_folding':

            eval_df = get_evaluation_results(
                gt_dir=f'{args.data_dir}/preprocessed',
                results_dir=f'{results_dir}/esmfold_pdb',
                mode='backbone',
                return_plddt=True
            )
            aar_df = cal_aar(
                gt_fasta=os.path.join(args.data_dir, 'aatype.fasta'),
                inv_fold_fasta=os.path.join(results_dir, 'aatype.fasta')
            )
            eval_df = eval_df.merge(aar_df, on='pdb_name')
            
            mean_rmsd = eval_df['rmsd'].mean()
            mean_tm = eval_df['tm_score'].mean()
            mean_aar = eval_df['aar'].mean()
            print(f'Mean RMSD: {mean_rmsd:.3f}, Mean TM-score: {mean_tm:.3f}, Mean AAR: {mean_aar:.3f}')
            eval_df.to_csv(os.path.join(results_dir, f'eval_results-rmsd={mean_rmsd:.3f}-tm={mean_tm:.3f}-aar={mean_aar:.3f}.csv'), index=False)
        
        else:
            raise NotImplementedError(f'Unknown task: {args.task}')

if __name__ == "__main__":
    main()