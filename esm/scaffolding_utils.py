# # NOTE: This file was modified from DPLM: https://github.com/bytedance/dplm/blob/main/src/byprot/utils/scaffold_utils.py

import random

import torch
import numpy as np
import pandas as pd

from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain

single_res = ["1qjg"]

# how long is the scaffold on the left side of the motif
scaffold_left = {
    "1PRW": [5, 20],
    "1BCF": [8, 15],
    "5TPN": [10, 40],
    "5IUS": [0, 30],
    "3IXT": [10, 40],
    "5YUI": [5, 30],
    "1QJG": [10, 20],
    "1YCR": [10, 40],
    "2KL8": [0, 0],
    "7MRX_60": [0, 38],
    "7MRX_85": [0, 63],
    "7MRX_128": [0, 122],
    "4JHW": [10, 25],
    "4ZYP": [10, 40],
    "5WN9": [10, 40],
    "5TRV_short": [0, 35],
    "5TRV_med": [0, 65],
    "5TRV_long": [0, 95],
    "6E6R_short": [0, 35],
    "6E6R_med": [0, 65],
    "6E6R_long": [0, 95],
    "6EXZ_short": [0, 35],
    "6EXZ_med": [0, 65],
    "6EXZ_long": [0, 95],
}

# how long is the scaffold on the right side of the motif
scaffold_right = {
    "1PRW": [5, 20],
    "1BCF": [8, 15],
    "5TPN": [10, 40],
    "5IUS": [0, 30],
    "3IXT": [10, 40],
    "5YUI": [10, 30],
    "1QJG": [10, 20],
    "1YCR": [10, 40],
    "2KL8": [0, 0],
    "7MRX_60": [0, 38],
    "7MRX_85": [0, 63],
    "7MRX_128": [0, 122],
    "4JHW": [10, 25],
    "4ZYP": [10, 40],
    "5WN9": [10, 40],
    "5TRV_short": [0, 35],
    "5TRV_med": [0, 65],
    "5TRV_long": [0, 95],
    "6E6R_short": [0, 35],
    "6E6R_med": [0, 65],
    "6E6R_long": [0, 95],
    "6EXZ_short": [0, 35],
    "6EXZ_med": [0, 65],
    "6EXZ_long": [0, 95],
}

# mapping the scaffolding task to the reference PDB code
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

scaffold_interval = {
    "1PRW": [[10, 25]],
    "1BCF": [[16, 30], [16, 30], [16, 30]],
    "5IUS": [[15, 40]],
    "5YUI": [[5, 20], [10, 35]],
    "1QJG": [[15, 30], [15, 30]],
    "2KL8": [[20, 20]],
    "4JHW": [[15, 30]],
}

total_length = {
    "1PRW": -1,
    "1BCF": -1,
    "5TPN": [50, 75],
    "5IUS": -1,
    "3IXT": [50, 75],
    "5YUI": [50, 100],
    "1QJG": -1,
    "1YCR": [40, 100],
    "2KL8": -1,
    "7MRX_60": [60, 60],
    "7MRX_85": [85, 85],
    "7MRX_128": [128, 128],
    "4JHW": [60, 90],
    "4ZYP": [30, 50],
    "5WN9": [35, 50],
    "5TRV_short": [56, 56],
    "5TRV_med": [86, 86],
    "5TRV_long": [116, 116],
    "6E6R_short": [48, 48],
    "6E6R_med": [78, 78],
    "6E6R_long": [108, 108],
    "6EXZ_short": [50, 50],
    "6EXZ_med": [80, 80],
    "6EXZ_long": [110, 110],
}

# if the motif has a single domain, len(start_idx_dict[pdb]) == 1
# if the motif have multi-domains, len(start_idx_dict[pdb]) > 1
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

# if the motif has a single domain, len(end_idx_dict[pdb]) == 1
# if the motif have multi-domains, len(end_idx_dict[pdb]) > 1
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

# given a list (Tensor) of non-masked residues, 
# get new start and end index for motif placed in scaffold.
def get_intervals(list, single_res_domain=False):

    if single_res_domain:
        start = [l.item() for l in list]
        stop = start
    else:
        start = []
        stop = []
        for i, item in enumerate(list):
            if i == 0:
                start.append(item.item())
            elif i == (len(list) - 1):
                stop.append(item.item())
            elif i != len(list) and (item + 1) != list[i + 1]:
                stop.append(item.item())
                start.append(list[i + 1].item())
    return start, stop

restype_with_x = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', \
                           'G', 'H', 'I', 'L', 'K', 'M', 'F', \
                            'P', 'S', 'T', 'W', 'Y', 'V', 'X'])

def get_motif(pdb_name, ori_pdb_name, pdb_file):

    start_idxs = start_idx_dict[ori_pdb_name]
    end_idxs = end_idx_dict[ori_pdb_name]
    assert len(start_idxs) == len(end_idxs)

    ref_pdb = ProteinChain.from_pdb(pdb_file)
    sequence = ref_pdb.sequence
    structure = ref_pdb.atom37_positions

    end_idxs = [i + 1 for i in end_idxs]
    motif_seq = list(sequence[start_idxs[0] : end_idxs[0]])
    motif_struct = torch.tensor(structure[start_idxs[0] : end_idxs[0]])
    
    # if len(spacer_list) == 0, we do not add any spacer
    for i in range(1, len(start_idxs)):
        interval_start = scaffold_interval[pdb_name][i-1][0]
        interval_end = scaffold_interval[pdb_name][i-1][1]
        spacer_num = random.randint(interval_start, interval_end)

        motif_seq += ["_"] * spacer_num
        # [L, 37, 3] -> [L + spacer, 37, 3] (padding with nan)
        motif_struct = torch.cat(
            # (motif_struct, torch.zeros((spacer_num, d_struct), dtype=motif_struct.dtype))
            (motif_struct, torch.full((spacer_num, 37, 3), float('nan'), dtype=motif_struct.dtype))
        )
        
        motif_seq += list(sequence[start_idxs[i] : end_idxs[i]])
        motif_struct = torch.cat(
            (motif_struct, torch.tensor(structure[start_idxs[i] : end_idxs[i]]))
        )
    
    # here we returen the motif sequence (still haven't been processed by the tokenizer) and structure
    return motif_seq, motif_struct


def get_initial_scaffolding_batches(pdb_file, pdb, ori_pdb, prot_num, device):
    
    # get motif tokens, and randomly insert mask tokens before and after the motif
    init_prot_list, scaffold_length_list = create_init_seq_struct(
        pdb, ori_pdb, pdb_file, prot_num
    )
    
    # create start and end indexes for the motif
    start_idxs_list, end_idxs_list = create_idxs_list(ori_pdb, init_prot_list)

    return init_prot_list, start_idxs_list, end_idxs_list, scaffold_length_list


def create_init_seq_struct(pdb, ori_pdb, pdb_file, num):

    init_prot_list = []
    scaffold_length_list = []

    (
        motif_seq, 
        motif_struct
    ) = get_motif(
            pdb_name=pdb,
            ori_pdb_name=ori_pdb,
            pdb_file=pdb_file
        )
    motif_overall_length = len(motif_seq)
    
    for i in range(num):
        
        length_compatible = False
        while length_compatible is False:
            scaffold_left_length = random.randint(
                scaffold_left[pdb][0], scaffold_left[pdb][1]
            )
            
            if total_length[pdb] != -1:
                current_length_range = [
                    scaffold_left_length
                    + motif_overall_length
                    + scaffold_right[pdb][0],
                    scaffold_left_length
                    + motif_overall_length
                    + scaffold_right[pdb][1],
                ]
                total_length_range = [
                    total_length[pdb][0],
                    total_length[pdb][1],
                ]
                length_range = [
                    max(current_length_range[0], total_length_range[0]),
                    min(current_length_range[1], total_length_range[1]),
                ]

                if length_range[0] > length_range[1]:
                    continue
                length_compatible = True
                scaffold_right_length = random.randint(
                    length_range[0], length_range[1]
                ) - (scaffold_left_length + motif_overall_length)
            
            else:
                length_compatible = True
                scaffold_right_length = random.randint(
                    scaffold_right[pdb][0], scaffold_right[pdb][1]
                )

            overall_length = (
                scaffold_left_length + motif_overall_length + scaffold_right_length
            )
            scaffold_length_list.append(scaffold_left_length + scaffold_right_length)

        # initialize the sequence track
        seq = (
            ["_"] * scaffold_left_length
            + motif_seq
            + ["_"] * scaffold_right_length
        )
        seq = "".join(seq)
        assert len(seq) == overall_length

        # initialize the structure track
        # [L, 37, 3] -> [scaffold_left + L + scaffold_right, 37, 3], padding with nan
        struct = torch.cat(
            (
                torch.full((scaffold_left_length, 37, 3), float('nan'), dtype=motif_struct.dtype),
                motif_struct,
                torch.full((scaffold_right_length, 37, 3), float('nan'), dtype=motif_struct.dtype),
            )
        )
        
        assert struct.shape[0] == overall_length
        init_prot_list.append(ESMProtein(sequence=seq, coordinates=struct))

    return init_prot_list, scaffold_length_list

def create_idxs_list(pdb, init_prot_list):

    single_res_domain = pdb in single_res

    start_idxs_list = []
    end_idxs_list = []
    get_intervals_seqs = [prot.sequence for prot in init_prot_list]

    for seq in get_intervals_seqs:
        
        nonmask_locations = torch.tensor([i for i, s in enumerate(seq) if s != "_"])
        new_start_idxs, new_end_idxs = get_intervals(
            nonmask_locations, single_res_domain=single_res_domain
        )
        start_idxs_list.append(new_start_idxs)
        end_idxs_list.append(new_end_idxs)

    return start_idxs_list, end_idxs_list

