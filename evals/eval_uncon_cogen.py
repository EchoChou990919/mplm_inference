import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--designability', action='store_true', help="enable designability evaluation (default: False)")
parser.add_argument('--sc_mode', type=str, default='backbone')
parser.add_argument('--diversity', action='store_true', help="enable diversity evaluation (default: False)")
parser.add_argument('--diversity_cluster_thrds', type=list, default=[0.5, 0.95])
parser.add_argument('--skip_inner_tm', action='store_true', help="skip the calculation of inner-tm score for diversity evaluation (default: False)")

parser.add_argument('--novelty', action='store_true', help="enable novelty evaluation (default: False)")
parser.add_argument('--pdb_or_sp', type=str, default='pdb')
parser.add_argument('--pdb_path', type=str, default='data/foldseek-pdb/pdb')
parser.add_argument('--swissprot_path', type=str, default='data/foldseek-swissprot/afdb')

parser.add_argument('--alignment_type', type=int, default=1, help='1 for tmalign, 2 for 3Di+AA')

parser.add_argument('--abc_ratio', action='store_true', help="enable alpha-beta-coil ratio evaluation (default: False)")

parser.add_argument('--results_root', type=str, default='')
parser.add_argument('--grid_eval', action='store_true', help="enable grid evaluation (default: False)")
parser.add_argument('--key_word', type=str, default='')
parser.add_argument('--exp_name', type=str, default='debug')
parser.add_argument('--prefix', type=str, default='sample')

parser.add_argument('--n_threads', type=int, default=12, help='Number of threads for parallel processing')

args = parser.parse_args()

import os
import re

import pandas as pd
import numpy as np

from biotite.structure.io.pdb import PDBFile
from biotite.structure import annotate_sse
from tqdm import tqdm
import subprocess
import multiprocessing as mp

from eval import get_pdb_file, cal_bb_rmsd, cal_bb_tm, extract_metrics_from_tmscore_output

def get_plddt_df(dir):

    plddt_df = pd.DataFrame(columns=['seq_len', 'seq_idx', 'plddt'])
    for pdb_folder_name in os.listdir(dir):
        folder_info = pdb_folder_name.split('_')
        seq_len = folder_info[-1]
        for pdb_file in os.listdir(os.path.join(dir, pdb_folder_name)):
            file_info = pdb_file.split('_')
            seq_idx = int(file_info[1])
            plddt = float(file_info[-1][:-4]) * 100
            plddt_df.loc[len(plddt_df)] = [seq_len, seq_idx, plddt]
    
    return plddt_df

def get_evaluation_results(esmfold_dir, salad_decoded_dir, mode='backbone', prefix='struct'):

    eval_df = pd.DataFrame(columns=['seq_len', 'seq_idx', 'rmsd', 'tm_score'])
    for folder_name in os.listdir(esmfold_dir):
        folder_info = folder_name.split('_')
        seq_len = folder_info[-1]

        for pdb_file in tqdm(os.listdir(os.path.join(esmfold_dir, folder_name))):
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
        
            eval_results['seq_len'] = seq_len
            eval_results['seq_idx'] = seq_idx
            eval_df.loc[len(eval_df)] = eval_results
    
    return eval_df

# TMalign: query pdb vs target pdb
def run_tmalign(query, target, fast=True):

    exec = "TMalign"
    cmd = f"{exec} {query} {target}"
    if fast:
        cmd += " -fast"
    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        return np.nan

    score_lines = []
    for line in output.decode().split("\n"):
        if line.startswith("TM-score"):
            score_lines.append(line)

    key_getter = lambda s: re.findall(r"Chain_[12]{1}", s)[0]
    score_getter = lambda s: float(re.findall(r"=\s+([0-9.]+)", s)[0])
    results_dict = {key_getter(s): score_getter(s) for s in score_lines}
    
    return results_dict["Chain_1"]

def tm_one2refs(
    query,
    targets,
    n_threads,
    fast = True,
    chunksize = 10
):
    args = [(query, target, fast) for target in targets]
    
    pool = mp.Pool(n_threads)
    values = list(pool.starmap(run_tmalign, args, chunksize=chunksize))
    pool.close()
    pool.join()
    
    return values

def foldseek_clustering(qt_dir, diversity_output_dir, div_aln_type, threshold):

    for dirname in tqdm(os.listdir(qt_dir)):
        if not os.path.isdir(os.path.join(qt_dir, dirname)):
            continue
        os.makedirs(os.path.join(diversity_output_dir, f'{dirname}_{threshold}'), exist_ok=True)
        cmd = f'foldseek easy-cluster {qt_dir}/{dirname} {diversity_output_dir}/{dirname}_{threshold}/res {diversity_output_dir}/{dirname}_{threshold} --alignment-type {div_aln_type} --cov-mode 0 --min-seq-id 0 --tmscore-threshold {threshold} > /dev/null 2>&1'
        os.system(cmd)
    
    return None

def tm_set2set(querys, targets, n_threads=args.n_threads):

    indexes = [q.split('/')[-1].split('.')[0] for q in querys]
    columns = [t.split('/')[-1].split('.')[0] for t in targets]
    tm_df = pd.DataFrame(columns=columns, index=indexes)
    for i, query_pdb in tqdm(enumerate(querys[:])):
        tm_scores = tm_one2refs(
            query_pdb,
            targets[i+1:],
            n_threads=n_threads
        )
        tm_df[query_pdb.split('/')[-1].split('.')[0]] = [0] * (i+1) + tm_scores
    
    return tm_df

def run_easy_search(cmd_body):
    cmd = f'foldseek easy-search {cmd_body} --exhaustive-search --tmscore-threshold 0.0 --format-output query,target,alntmscore,qtmscore,ttmscore > /dev/null 2>&1'
    os.system(cmd)
    return None

def get_novelty_df(novelty_output_dir):

    alntmsocre_novelty_df = pd.DataFrame(columns=['group', 'query', 'target', 'tmscore'])
    qtmscore_novelty_df = pd.DataFrame(columns=['group', 'query', 'target', 'tmscore'])
    ttmscore_novelty_df = pd.DataFrame(columns=['group', 'query', 'target', 'tmscore'])
    for dirname in os.listdir(novelty_output_dir):
        novelty_output_path = os.path.join(novelty_output_dir, dirname)
        if not os.path.isdir(novelty_output_path):
            continue
    
        for tsv_file in os.listdir(novelty_output_path):
            if not tsv_file.endswith('.tsv'):
                continue
            
            try:
                tmp_df = pd.read_csv(os.path.join(novelty_output_path, tsv_file), sep='\t', header=None)
                if tmp_df.empty:
                    alntmsocre_novelty_df.loc[len(alntmsocre_novelty_df)] = [dirname, tsv_file[:-4], 'None', 0]
                    qtmscore_novelty_df.loc[len(qtmscore_novelty_df)] = [dirname, tsv_file[:-4], 'None', 0]
                    ttmscore_novelty_df.loc[len(ttmscore_novelty_df)] = [dirname, tsv_file[:-4], 'None', 0]
                
                else:
                    tmp_df.columns = ['query', 'target', 'alntmscore', 'qtmscore', 'ttmscore']
                    # record the max tmscore
                    max_alntmscore_idx = tmp_df['alntmscore'].idxmax()
                    alntmsocre_novelty_df.loc[len(alntmsocre_novelty_df)] = [dirname, tmp_df.loc[max_alntmscore_idx, 'query'], tmp_df.loc[max_alntmscore_idx, 'target'], tmp_df.loc[max_alntmscore_idx, 'alntmscore']]
                    max_qtmscore_idx = tmp_df['qtmscore'].idxmax()
                    qtmscore_novelty_df.loc[len(qtmscore_novelty_df)] = [dirname, tmp_df.loc[max_qtmscore_idx, 'query'], tmp_df.loc[max_qtmscore_idx, 'target'], tmp_df.loc[max_qtmscore_idx, 'qtmscore']]
                    max_ttmscore_idx = tmp_df['ttmscore'].idxmax()
                    ttmscore_novelty_df.loc[len(ttmscore_novelty_df)] = [dirname, tmp_df.loc[max_ttmscore_idx, 'query'], tmp_df.loc[max_ttmscore_idx, 'target'], tmp_df.loc[max_ttmscore_idx, 'ttmscore']]

            except Exception as e:
                print(f'Error in {dirname}/{tsv_file}: {e}')
                alntmsocre_novelty_df.loc[len(alntmsocre_novelty_df)] = [dirname, tsv_file[:-4], 'None', 0]
                qtmscore_novelty_df.loc[len(qtmscore_novelty_df)] = [dirname, tsv_file[:-4], 'None', 0]
                ttmscore_novelty_df.loc[len(ttmscore_novelty_df)] = [dirname, tsv_file[:-4], 'None', 0]
                
    return alntmsocre_novelty_df, qtmscore_novelty_df, ttmscore_novelty_df

def get_sse_structure(pdb_file_path):

    pdb_file = PDBFile.read(pdb_file_path)
    structure = pdb_file.get_structure()[0]
    
    sse = annotate_sse(structure)
    alpha_ratio = (sse == 'a').sum() / len(sse)
    beta_ratio = (sse == 'b').sum() / len(sse)
    coil_ratio = (sse == 'c').sum() / len(sse)
    
    return alpha_ratio, beta_ratio, coil_ratio

def main():
    
    root = args.results_root
    if args.grid_eval:
        grid_list = os.listdir(root)
    else:
        grid_list = ['']
    for grid in grid_list:
        if args.key_word not in grid:
            continue
        if args.exp_name in grid:
            continue
        print(f'Processing grid: {grid}')

        results_dir = os.path.join(root, grid)
        qt_dir = f'{results_dir}/decoded_pdb'

        eval_dir = os.path.join(root, grid, 'eval_results')
        os.makedirs(eval_dir, exist_ok=True)
        
        # for designability
        if args.designability:
            if args.sc_mode == 'backbone':
                eval_file = [file for file in os.listdir(eval_dir) if file.startswith('designability') & file.endswith('csv') & ('-bb_' in file)]
            elif args.sc_mode == 'TMscore':
                eval_file = [file for file in os.listdir(eval_dir) if file.startswith('designability') & file.endswith('csv') & ~('-bb_' in file)]
            else:
                raise NotImplementedError(f'Unsupported mode: {args.sc_mode}')
            
            if len(eval_file) == 0:
                print('Calculating designability...')
                # get plddt
                plddt_df = get_plddt_df(f'{results_dir}/esmfold_pdb')
                # get sc_rmsd and sc_tm
                eval_df = get_evaluation_results(f'{results_dir}/esmfold_pdb', f'{results_dir}/decoded_pdb', mode=args.sc_mode, prefix=args.prefix)
                combined_df = pd.merge(plddt_df, eval_df, on=['seq_len', 'seq_idx'])
                combined_df = combined_df.sort_values(['seq_len', 'seq_idx'])

                mean_plddt = plddt_df['plddt'].mean()
                mean_rmsd = eval_df['rmsd'].mean()
                mean_tm = eval_df['tm_score'].mean()
                if args.sc_mode == 'backbone':
                    designablity_file = f'designability-plddt={mean_plddt:.3f}-bb_rmsd={mean_rmsd:.3f}-bb_tm={mean_tm:.3f}.csv'
                elif args.sc_mode == 'TMscore':
                    designablity_file = f'designability-plddt={mean_plddt:.3f}-rmsd={mean_rmsd:.3f}-tm={mean_tm:.3f}.csv'
                else:
                    raise NotImplementedError(f'Unsupported mode: {args.sc_mode}')
                
                print(f'pLDDT: {mean_plddt:.5f}, RMSD: {mean_rmsd:.5f}, TM-score: {mean_tm:.5f}')
                combined_df.to_csv(f'{results_dir}/eval_results/{designablity_file}', index=False)
                combined_df_by_length = combined_df.groupby('seq_len').mean(numeric_only=True).reset_index()
                combined_df_by_length.to_csv(f'{results_dir}/eval_results/length_wise_designability.csv', index=False)

        # for diversity
        if args.diversity:
            eval_file = [file for file in os.listdir(eval_dir) if file.startswith(f'diversity-aln_type={args.alignment_type}') and file.endswith('txt')]
            if len(eval_file) == 0:
                print('Calculating diversity...')
                diversity_output_dir = f'{results_dir}/eval_results/diversity-aln_type={args.alignment_type}'
                os.makedirs(diversity_output_dir, exist_ok=True)
                
                # get inner TM-score
                if not args.skip_inner_tm:
                    mean_inner_tm = []
                    for dirname in os.listdir(qt_dir):
                        if not os.path.isdir(os.path.join(qt_dir, dirname)):
                            continue
                        qt_pdb_dir = os.path.join(qt_dir, dirname)
                        qt_paths = [os.path.join(qt_pdb_dir, pdb_name) for pdb_name in os.listdir(qt_pdb_dir)]

                        tm_df = tm_set2set(qt_paths, qt_paths).fillna(0)
                        tm_scores = tm_df.values
                        mean_tm_scores = tm_scores.sum() * 2 / (tm_scores.shape[0] * (tm_scores.shape[1] - 1))
                        mean_inner_tm.append(mean_tm_scores)
                        tm_df.to_csv(os.path.join(diversity_output_dir, f'{dirname}-tm={mean_tm_scores:.3f}.csv'), index=True)
                    mean_inner_tm = np.array(mean_inner_tm).mean()
                else:
                    mean_inner_tm = 0.0
                print(f'Inner TM-score: {mean_inner_tm:.3f}')
                
                # get max cluster at different thresholds
                max_cluster_at_thrds = []
                for threshold in args.diversity_cluster_thrds:
                    foldseek_clustering(qt_dir, diversity_output_dir, args.alignment_type, threshold)

                    num_cluster = 0
                    num_member = 0
                    for dirname in os.listdir(diversity_output_dir):
                        if not os.path.isdir(os.path.join(diversity_output_dir, dirname)):
                            continue
                        if not dirname.endswith(f'_{threshold}'):
                            continue
                        cluster_df = pd.read_csv(f'{diversity_output_dir}/{dirname}/res_cluster.tsv', sep='\t', header=None)
                        cluster_df.columns = ['repr_id', 'member_id']
                        tmp_num_cluster = cluster_df['repr_id'].nunique()
                        tmp_num_member = cluster_df['member_id'].nunique()
                        num_cluster += tmp_num_cluster
                        num_member += tmp_num_member
                        with open(os.path.join(f'{diversity_output_dir}/{dirname}_num_cluster={tmp_num_cluster}.txt'), 'w') as f:
                            f.write(f'Number of clusters: {tmp_num_cluster}, Number of members: {tmp_num_member}.\n')
                    max_cluster = num_cluster / num_member
                    
                    print(f'Threshold: {threshold}, Max cluster: {max_cluster:.3f}')
                    max_cluster_at_thrds.append(f'cluster_{threshold}={max_cluster:.3f}')
                
                max_cluster_string = '-'.join(max_cluster_at_thrds)
                with open(os.path.join(f'{results_dir}/eval_results/', f'diversity-aln_type={args.alignment_type}-inner_tm={mean_inner_tm:.3f}-{max_cluster_string}.txt'), 'w') as f:
                    f.write(f'Inner-TM: {mean_inner_tm:.3f}, Max cluster at different thresholds: {max_cluster_at_thrds}.\n')
        
        # for novelty
        if args.novelty:
            for pdb_or_sp, database_path in [('pdb', args.pdb_path), ('sp', args.swissprot_path)]:
                eval_file = [file for file in os.listdir(eval_dir) if file.startswith(f'novelty-aln_type={args.alignment_type}-{args.pdb_or_sp}') & file.endswith('csv')]
                if len(eval_file) > 0:
                    continue
                novelty_output_dir = f'{results_dir}/eval_results/novelty-aln_type={args.alignment_type}-{pdb_or_sp}'
                os.makedirs(novelty_output_dir, exist_ok=True)

                # get pdb tm-score
                cmd_bodies = []
                for dirname in os.listdir(qt_dir):
                    if not os.path.isdir(os.path.join(qt_dir, dirname)):
                        continue
                    novelty_output_path = os.path.join(novelty_output_dir, dirname)
                    os.makedirs(novelty_output_path, exist_ok=True)
                    qt_pdb_dir = os.path.join(qt_dir, dirname)

                    for pdb_name in os.listdir(qt_pdb_dir):
                        pdb_file_path = os.path.join(qt_pdb_dir, pdb_name)
                        cmd_bodies.append(f'{pdb_file_path} {database_path} {novelty_output_path}/{pdb_name[:-4]}.tsv {novelty_output_dir}/tmp --alignment-type {args.alignment_type}')
                with mp.Pool(args.n_threads) as pool:
                    _ = list(tqdm(pool.imap(run_easy_search, cmd_bodies), total=len(cmd_bodies)))
                
                aln_novelty_df, qt_novelty_df, tt_novelty_df = get_novelty_df(novelty_output_dir)
                mean_aln_tm = aln_novelty_df['tmscore'].mean()
                mean_qt_tm = qt_novelty_df['tmscore'].mean()
                mean_tt_tm = tt_novelty_df['tmscore'].mean()
                
                print(f'{pdb_or_sp} aln tm-score: {mean_aln_tm:.3f}, qt tm-score: {mean_qt_tm:.3f}, tt tm-score: {mean_tt_tm:.3f}')
                aln_novelty_df.to_csv(f'{results_dir}/eval_results/novelty-aln_type={args.alignment_type}-{pdb_or_sp}_aln_tm={mean_aln_tm:.3f}.csv', index=False)
                qt_novelty_df.to_csv(f'{results_dir}/eval_results/novelty-aln_type={args.alignment_type}-{pdb_or_sp}_qt_tm={mean_qt_tm:.3f}.csv', index=False)
                tt_novelty_df.to_csv(f'{results_dir}/eval_results/novelty-aln_type={args.alignment_type}-{pdb_or_sp}_tt_tm={mean_tt_tm:.3f}.csv', index=False)

                os.system(f"rm -rf {novelty_output_dir}")

        # for alpha-beta-coil ratio
        if args.abc_ratio:
            eval_file = [file for file in os.listdir(eval_dir) if file.startswith('abc_ratio') & file.endswith('csv')]
            if len(eval_file) == 0:
                print('Calculating alpha-beta-coil ratio...')
                
                abc_df = pd.DataFrame(columns=['group', 'prot', 'alpha_ratio', 'beta_ratio', 'coil_ratio'])
                for dirname in os.listdir(qt_dir):
                    if not os.path.isdir(os.path.join(qt_dir, dirname)):
                        continue
                    qt_pdb_dir = os.path.join(qt_dir, dirname)
                    for pdb_file in os.listdir(qt_pdb_dir):
                        pdb_file_path = os.path.join(qt_pdb_dir, pdb_file)
                        alpha_ratio, beta_ratio, coil_ratio = get_sse_structure(pdb_file_path)
                        abc_df.loc[len(abc_df)] = [dirname, pdb_file[:-4], alpha_ratio, beta_ratio, coil_ratio]
                
                mean_alpha_ratio = abc_df['alpha_ratio'].mean()
                mean_beta_ratio = abc_df['beta_ratio'].mean()
                mean_coil_ratio = abc_df['coil_ratio'].mean()
                
                print(f'Alpha: {mean_alpha_ratio:.3f}, Beta: {mean_beta_ratio:.3f}, Coil: {mean_coil_ratio:.3f}')
                abc_df.to_csv(f'{results_dir}/eval_results/abc_ratio={mean_alpha_ratio:.3f}-{mean_beta_ratio:.3f}-{mean_coil_ratio:.3f}.csv', index=False)
        

if __name__ == "__main__":
    main()