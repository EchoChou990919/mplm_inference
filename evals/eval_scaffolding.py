import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--results_root', type=str, default='')

parser.add_argument('--sc_mode', type=str, default='backbone', choices=['backbone', 'TMscore'], help='The mode for self-consistency evaluation.')
parser.add_argument('--sc_motif_mode', type=str, default='concat', choices=['concat', 'separ'], help='The mode for motif RMSD evaluation.')

parser.add_argument('--tm_thr', type=float, default=0.8, help='The TM-score threshold for successful scaffolding.')
parser.add_argument('--motif_rmsd_thr', type=float, default=1.0, help='The motif RMSD threshold for successful scaffolding.')

parser.add_argument('--diversity', action='store_true', help="enable diversity evaluation via foldseek clustering (default: False)")
parser.add_argument('--diversity_cluster_thrd', type=float, default=0.5, help='TM-score threshold for foldseek clustering.')
parser.add_argument('--aln_type', type=int, default=1, help='Alignment type for foldseek easy-cluster. 1 for tmalign, 2 for 3Di+AA.')

parser.add_argument('--grid_eval', action='store_true', help="enable grid evaluation (default: False)")
parser.add_argument('--key_word', type=str, default='')
parser.add_argument('--prefix', type=str, default='sample', help='The prefix of decoded pdb files.')
parser.add_argument(
    '--allow_motif_seq_mismatch',
    action='store_true',
    help=('Warnings instead of asserts when motif sequence mismatch happens.'),
)

args = parser.parse_args()

import os
import subprocess

import pandas as pd
import numpy as np
from tqdm import tqdm

from eval import get_pdb_file, cal_bb_rmsd, cal_bb_tm, extract_metrics_from_tmscore_output

motif_name_mapping = {
    "1PRW": "1prw",
    "1BCF": "1bcf",
    "5TPN": "5tpn",
    "5IUS": "5ius",
    "3IXT": "3ixt",
    "5YUI": "5yui",
    "1QJG": "1qjg",
    "1YCR": "1ycr",
    "2KL8": "2kl8",
    "7MRX_60": "7mrx",
    "7MRX_85": "7mrx",
    "7MRX_128": "7mrx",
    "4JHW": "4jhw",
    "4ZYP": "4zyp",
    "5WN9": "5wn9",
    "5TRV_short": "5trv",
    "5TRV_med": "5trv",
    "5TRV_long": "5trv",
    "6E6R_short": "6e6r",
    "6E6R_med": "6e6r",
    "6E6R_long": "6e6r",
    "6EXZ_short": "6exz",
    "6EXZ_med": "6exz",
    "6EXZ_long": "6exz",
}

start_idx_dict = {
    "1prw": [15, 51],
    "1bcf": [90, 122, 46, 17],
    "5tpn": [108],
    "3ixt": [0],
    "4jhw": [144, 37],
    "4zyp": [357],
    "5wn9": [1],
    "5ius": [88, 34],
    "5yui": [89, 114, 194],
    "6vw1": [5, 45],
    "1qjg": [37, 13, 98],
    "1ycr": [2],
    "2kl8": [0, 27],
    "7mrx": [25],
    "5trv": [45],
    "6e6r": [22],
    "6exz": [25],
}

end_idx_dict = {
    "1prw": [34, 70],
    "1bcf": [98, 129, 53, 24],
    "5tpn": [126],
    "3ixt": [23],
    "4jhw": [159, 43],
    "4zyp": [371],
    "5wn9": [20],
    "5ius": [109, 53],
    "5yui": [93, 116, 196],
    "6vw1": [23, 63],
    "1qjg": [37, 13, 98],
    "1ycr": [10],
    "2kl8": [6, 78],
    "7mrx": [46],
    "5trv": [69],
    "6e6r": [34],
    "6exz": [39],
}

def get_plddt_df(dir):
    plddt_df = pd.DataFrame(columns=['pdb_name', 'seq_idx', 'plddt'])
    for pdb_folder_name in os.listdir(dir):
        for pdb_file in os.listdir(os.path.join(dir, pdb_folder_name)):
            file_info = pdb_file.split('_')
            seq_idx = int(file_info[1])
            plddt = float(file_info[-1][:-4]) * 100
            plddt_df.loc[len(plddt_df)] = [pdb_folder_name, seq_idx, plddt]
    return plddt_df

def get_evaluation_results(esmfold_dir, salad_decoded_dir, mode='backbone', prefix='struct'):
    
    eval_df = pd.DataFrame(columns=['pdb_name', 'seq_idx', 'rmsd', 'tm_score'])
    for folder_name in tqdm(os.listdir(esmfold_dir)):
        for pdb_file in os.listdir(os.path.join(esmfold_dir, folder_name)):
            file_info = pdb_file.split('_')
            seq_idx = int(file_info[1])
            esmfold_struct = os.path.join(esmfold_dir, folder_name, pdb_file)
            decoded_struct = os.path.join(salad_decoded_dir, folder_name, f'{prefix}_{seq_idx}.pdb')
            
            if mode == 'backbone':
                predicted_bb_pos, _, _ = get_pdb_file(esmfold_struct, mode='backbone')
                generated_bb_pos, seq, _ = get_pdb_file(decoded_struct, mode='backbone')
                rmsd = cal_bb_rmsd(generated_bb_pos, predicted_bb_pos)
                tm_score = cal_bb_tm(generated_bb_pos, predicted_bb_pos, seq)
                eval_results = {
                    'rmsd': rmsd,
                    'tm_score': tm_score
                }
            elif mode == 'TMscore':
                tm_output = subprocess.run(['TMscore', decoded_struct, esmfold_struct], capture_output=True, text=True).stdout
                eval_results = extract_metrics_from_tmscore_output(tm_output)
            else:
                raise NotImplementedError(f'Unsupported mode: {mode}')
            
            eval_results['pdb_name'] = folder_name
            eval_results['seq_idx'] = seq_idx
            eval_df.loc[len(eval_df)] = eval_results
    
    return eval_df

def get_motif_rmsd(scaffolding_data_dir, results_dir, motif_name_mapping, sc_motif_mode='concat', prefix='struct', allow_motif_seq_mismatch=False):
    
    eval_df = pd.DataFrame(columns=['pdb_name', 'seq_idx', 'pred_motif_rmsd', 'gen_motif_rmsd'])
    esmfold_dir = os.path.join(results_dir, 'esmfold_pdb')
    decoded_dir = os.path.join(results_dir, 'decoded_pdb')
    for pdb_name in os.listdir(esmfold_dir):
        assert pdb_name in motif_name_mapping, f'Folder {pdb_name} not found in motif_name_mapping'
        
        ori_motif = os.path.join(scaffolding_data_dir, 'ori_pdbs', f'{motif_name_mapping[pdb_name]}_reference.pdb')
        ori_motif_pos, ori_motif_seq, _ = get_pdb_file(ori_motif, mode='backbone')
        motif_start_list = start_idx_dict[motif_name_mapping[pdb_name]]
        motif_end_list = end_idx_dict[motif_name_mapping[pdb_name]]
        
        if sc_motif_mode == 'concat':
            gt_motif_pos = ori_motif_pos[motif_start_list[0]:motif_end_list[0]+1]
            gt_motif_seq = ori_motif_seq[motif_start_list[0]:motif_end_list[0]+1]
            for s, e in zip(motif_start_list[1:], motif_end_list[1:]):
                gt_motif_pos = np.concatenate((gt_motif_pos, ori_motif_pos[s:e+1]), axis=0)
                gt_motif_seq = gt_motif_seq + ori_motif_seq[s:e+1]
        elif sc_motif_mode == 'separ':
            gt_motif_pos = []
            gt_motif_seq = []
            for s, e in zip(motif_start_list, motif_end_list):
                gt_motif_pos.append(ori_motif_pos[s:e+1])
                gt_motif_seq.append(ori_motif_seq[s:e+1])
        else:
            raise NotImplementedError(f'Unsupported sc_motif_mode: {sc_motif_mode}')
        
        try:
            for pdb_file in os.listdir(os.path.join(esmfold_dir, pdb_name)):
                file_info = pdb_file.split('_')
                seq_idx = int(file_info[1])
                esmfold_struct = os.path.join(esmfold_dir, pdb_name, pdb_file)
                decoded_struct = os.path.join(decoded_dir, pdb_name, f'{prefix}_{seq_idx}.pdb')
                
                predicted_pos, predicted_seq, _ = get_pdb_file(esmfold_struct, mode='backbone')
                generated_pos, generated_seq, _ = get_pdb_file(decoded_struct, mode='backbone')
                
                start_end_dict = np.load(os.path.join(results_dir, 'start_end_scaffold', f'{pdb_name}.npz'))
                start_list = start_end_dict['start_idxs_list'][seq_idx]
                end_list = start_end_dict['end_idxs_list'][seq_idx]

                if sc_motif_mode == 'concat':
                    
                    predicted_motif_pos = predicted_pos[start_list[0]:end_list[0]+1]
                    predicted_motif_seq = predicted_seq[start_list[0]:end_list[0]+1]
                    generated_motif_pos = generated_pos[start_list[0]:end_list[0]+1]
                    generated_motif_seq = generated_seq[start_list[0]:end_list[0]+1]
                    for s, e in zip(start_list[1:], end_list[1:]):
                        predicted_motif_pos = np.concatenate((predicted_motif_pos, predicted_pos[s:e+1]), axis=0)
                        predicted_motif_seq = predicted_motif_seq + predicted_seq[s:e+1]
                        generated_motif_pos = np.concatenate((generated_motif_pos, generated_pos[s:e+1]), axis=0)
                        generated_motif_seq = generated_motif_seq + generated_seq[s:e+1]
                    if gt_motif_seq != predicted_motif_seq:
                        msg = f'{gt_motif_seq} != {predicted_motif_seq} for {pdb_name} at seq_idx {seq_idx}'
                        if allow_motif_seq_mismatch:
                            print(f'[motif-seq mismatch, predicted] {msg}')
                        else:
                            assert False, msg
                    if gt_motif_seq != generated_motif_seq:
                        msg = f'{gt_motif_seq} != {generated_motif_seq} for {pdb_name} at seq_idx {seq_idx}'
                        if allow_motif_seq_mismatch:
                            print(f'[motif-seq mismatch, generated] {msg}')
                        else:
                            assert False, msg
                    
                    pred_motif_rmsd = cal_bb_rmsd(predicted_motif_pos, gt_motif_pos)
                    gen_motif_rmsd = cal_bb_rmsd(generated_motif_pos, gt_motif_pos)
                
                elif sc_motif_mode == 'separ':

                    predicted_motif_pos = []
                    predicted_motif_seq = []
                    generated_motif_pos = []
                    generated_motif_seq = []
                    for s, e in zip(start_list, end_list):
                        predicted_motif_pos.append(predicted_pos[s:e+1])
                        predicted_motif_seq.append(predicted_seq[s:e+1])
                        generated_motif_pos.append(generated_pos[s:e+1])
                        generated_motif_seq.append(generated_seq[s:e+1])

                    pred_motif_rmsd = []
                    gen_motif_rmsd = []
                    for i in range(len(gt_motif_pos)):
                        if gt_motif_seq[i] != predicted_motif_seq[i]:
                            msg = f'{gt_motif_seq[i]} != {predicted_motif_seq[i]} for {pdb_name} at seq_idx {seq_idx}'
                            if allow_motif_seq_mismatch:
                                print(f'[motif-seq mismatch, predicted seg {i}] {msg}')
                            else:
                                assert False, msg
                        if gt_motif_seq[i] != generated_motif_seq[i]:
                            msg = f'{gt_motif_seq[i]} != {generated_motif_seq[i]} for {pdb_name} at seq_idx {seq_idx}'
                            if allow_motif_seq_mismatch:
                                print(f'[motif-seq mismatch, generated seg {i}] {msg}')
                            else:
                                assert False, msg
                        pred_motif_rmsd.append(cal_bb_rmsd(predicted_motif_pos[i], gt_motif_pos[i]))
                        gen_motif_rmsd.append(cal_bb_rmsd(generated_motif_pos[i], gt_motif_pos[i]))
                                    
                    pred_motif_rmsd = np.mean(pred_motif_rmsd)
                    gen_motif_rmsd = np.mean(gen_motif_rmsd)
                
                else:
                    raise NotImplementedError(f'Unsupported sc_motif_mode: {sc_motif_mode}')
                
                eval_df.loc[len(eval_df)] = {
                    'pdb_name': pdb_name,
                    'seq_idx': seq_idx,
                    'pred_motif_rmsd': pred_motif_rmsd,
                    'gen_motif_rmsd': gen_motif_rmsd
                }
        
        except Exception as e:
            print(f'Error processing {pdb_name}: {e}')
            continue
    
    return eval_df

def get_diversity_results(decoded_dir, eval_dir, threshold=0.5):
    """Run foldseek easy-cluster per scaffolding problem and return cluster counts."""
    diversity_dir = os.path.join(eval_dir, f'diversity-aln_type={args.aln_type}-thrd={threshold}')
    os.makedirs(diversity_dir, exist_ok=True)

    rows = []
    for pdb_name in sorted(os.listdir(decoded_dir)):
        pdb_dir = os.path.join(decoded_dir, pdb_name)
        if not os.path.isdir(pdb_dir):
            continue
        tmp_dir = os.path.join(diversity_dir, pdb_name)
        os.makedirs(tmp_dir, exist_ok=True)

        cmd = (
            f'foldseek easy-cluster {pdb_dir} {tmp_dir}/res {tmp_dir} '
            f'--alignment-type {args.aln_type} --cov-mode 0 --min-seq-id 0 '
            f'--tmscore-threshold {threshold} --single-step-clustering '
            f'> /dev/null 2>&1'
        )
        os.system(cmd)

        cluster_tsv = os.path.join(tmp_dir, 'res_cluster.tsv')
        if os.path.exists(cluster_tsv):
            cluster_df = pd.read_csv(cluster_tsv, sep='\t', header=None, names=['repr_id', 'member_id'])
            n_clusters = cluster_df['repr_id'].nunique()
            n_members = cluster_df['member_id'].nunique()
        else:
            n_clusters = 0
            n_members = 0
        rows.append({'pdb_name': pdb_name, 'n_clusters': n_clusters, 'n_members': n_members})
        print(f'  {pdb_name}: {n_clusters} clusters / {n_members} samples')

    diversity_df = pd.DataFrame(rows)
    return diversity_df


def main():

    root = args.results_root
    pdb_name_list = sorted(list(motif_name_mapping.keys()))
    if args.grid_eval:
        grid_list = os.listdir(root)
    else:
        grid_list = ['']
    for grid in grid_list:
        if args.key_word not in grid:
            continue
        print(f'Processing grid: {root}/{grid}')

        results_dir = os.path.join(root, grid)

        eval_dir = os.path.join(root, grid, 'eval_results')
        os.makedirs(eval_dir, exist_ok=True)

        # --- Diversity evaluation ---
        if args.diversity:
            thrd = args.diversity_cluster_thrd
            div_file = [f for f in os.listdir(eval_dir) if f.startswith(f'diversity-aln_type={args.aln_type}-cluster_{thrd}') and f.endswith('.csv')]
            if len(div_file) == 0:
                print('Calculating diversity...')
                diversity_df = get_diversity_results(f'{results_dir}/decoded_pdb', eval_dir, threshold=thrd)
                total_clusters = diversity_df['n_clusters'].sum()
                total_members = diversity_df['n_members'].sum()
                mean_ratio = (diversity_df['n_clusters'] / diversity_df['n_members'].clip(lower=1)).mean()
                print(f'Diversity: total_clusters={total_clusters}, total_members={total_members}, mean_cluster_ratio={mean_ratio:.3f}')
                diversity_df.to_csv(os.path.join(eval_dir, f'diversity-aln_type={args.aln_type}-cluster_{thrd}={mean_ratio:.3f}.csv'), index=False)
            else:
                print(f'Diversity already computed: {div_file[0]}')
            continue

        # --- Standard evaluation ---
        eval_file = [file for file in os.listdir(eval_dir) if file.startswith(f'eval_results') & file.endswith('csv')]
        if len(eval_file) > 0:
            continue
        
        try:
            plddt_df = get_plddt_df(f'{results_dir}/esmfold_pdb')
            eval_df = get_evaluation_results(f'{results_dir}/esmfold_pdb', f'{results_dir}/decoded_pdb', mode=args.sc_mode, prefix=args.prefix)
            eval_df = pd.merge(plddt_df, eval_df, on=['pdb_name', 'seq_idx'])
            motif_eval_df = get_motif_rmsd(args.data_dir, results_dir, motif_name_mapping, sc_motif_mode=args.sc_motif_mode, prefix=args.prefix, allow_motif_seq_mismatch=args.allow_motif_seq_mismatch)
            eval_df = pd.merge(eval_df, motif_eval_df, on=['pdb_name', 'seq_idx'], how='outer')
            our_result = eval_df[(eval_df['tm_score'] > args.tm_thr) & (eval_df['pred_motif_rmsd'] < args.motif_rmsd_thr)]
        
        except Exception as e:
            print(f'Error processing grid {grid}: {e}')
            continue

        pass_num = 0
        success_rate = 0
        for pdb_name in pdb_name_list:
            tmp_num = len(our_result[our_result['pdb_name'] == pdb_name])
            if tmp_num > 0:
                pass_num += 1
            success_rate += tmp_num
            print(f'{pdb_name}, {tmp_num}')
        success_rate /= len(eval_df)
        
        print(f'Pass num: {pass_num}/{len(pdb_name_list)}; Success rate: {success_rate*100:.3f}%')
        os.makedirs(f'{results_dir}/eval_results/', exist_ok=True)
        eval_df.to_csv(f'{results_dir}/eval_results/eval_results-pass_num={pass_num}-success_rate={success_rate:.3f}.csv', index=False)

if __name__ == "__main__":
    main()