import os
import re

import numpy as np
import pandas as pd
from numpy.linalg import svd

from biotite.structure.io.pdb import PDBFile
from biotite.structure import get_residues

from tmtools import tm_align

import pickle

# Kabsch for rigid alignment
def kabsch_align(P, Q):

    # compute centroids
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid
    
    # compute covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD decomposition
    U, S, Vt = svd(H)
    R = Vt.T @ U.T
    
    # process reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # compute translation
    t = Q_centroid - R @ P_centroid
    
    return R, t

def index_align(x, y, index=None):
    # align y to x
    
    x = np.asarray(x)
    y = np.asarray(y)
    if index is None:
        index = np.zeros(len(x), dtype=int)
    else:
        index = np.asarray(index)
    
    aligned = np.zeros_like(y)
    
    for batch_id in np.unique(index):
        mask = (index == batch_id)
        P = x[mask]
        Q = y[mask]
        
        if len(P) < 3:
            aligned[mask] = Q
            continue
            
        R, t = kabsch_align(P, Q)
        aligned[mask] = (P @ R.T) + t
        
    return aligned

def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape, f"Input matrices must have the same dimensions: {A.shape} vs {B.shape}"
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}

def get_coord_from_residue(structure, res_id, atom_name):
    coord = structure[(structure.res_id == res_id) & (structure.atom_name == atom_name)].coord
    if len(coord) == 0:
        return (0, 0, 0), False
    else:
        return coord[0], True

def get_pdb_file(file_path, mode='backbone', include_o=False):
    
    pdb_file = PDBFile.read(file_path)
    structure = pdb_file.get_structure()[0]
    
    if mode == 'ca_only':
        coord = structure[structure.atom_name == 'CA'].coord # (N, 3)
        mask = np.array([True] * len(coord))
    
    elif mode == 'backbone':
        res_ids = structure[structure.atom_name == 'CA'].res_id
        coord, mask = [], []
        for res_id in res_ids:
            n_coord, n_mask = get_coord_from_residue(structure, res_id, 'N')
            ca_coord, ca_mask = get_coord_from_residue(structure, res_id, 'CA')
            c_coord, c_mask = get_coord_from_residue(structure, res_id, 'C')
            if include_o:
                o_coord, o_mask = get_coord_from_residue(structure, res_id, 'O')
                coord.append([n_coord, ca_coord, c_coord, o_coord])
                mask.append([n_mask, ca_mask, c_mask, o_mask])
            else:
                coord.append([n_coord, ca_coord, c_coord])
                mask.append([n_mask, ca_mask, c_mask])

        coord = np.array(coord) # (N, 3, 3) or (N, 4, 3)
        mask = np.array(mask).astype(bool)
    
    else:
        raise NotImplementedError(f'Unknown mode: {mode}')
    
    _, residues = get_residues(structure)
    seq = ''
    for res in residues:
        if res in restype_3to1:
            seq += restype_3to1[res]
        else:
            seq += 'X'
    
    return coord, seq, mask

restype_with_x = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', \
                           'G', 'H', 'I', 'L', 'K', 'M', 'F', \
                           'P', 'S', 'T', 'W', 'Y', 'V', 'X'])

def get_pkl_file(file_path, mode='backbone', include_o=False):

    with open(file_path, "rb") as f:
        pkl_file = pickle.load(f)

    # mask for valid residues
    aatype = pkl_file['aatype']
    seq_mask = (aatype < 20)
    seq_mask_idx = [i for i, m in enumerate(seq_mask) if m]
    seq_mask[seq_mask_idx[0]: seq_mask_idx[-1]+1] = True  # make the mask continuous

    atom37_coords = pkl_file['atom_positions']
    if mode == 'ca_only':
        coord = atom37_coords[:, 1, :][seq_mask]
    elif mode == 'backbone':
        coord = atom37_coords[:, :3, :][seq_mask]
        if include_o:
            coord = np.concatenate([coord, atom37_coords[:, 4:5, :][seq_mask]], axis=1)
    else:
        raise NotImplementedError(f'Unknown mode: {mode}')
    
    aatype = pkl_file['aatype'][seq_mask]
    seq = ''.join(restype_with_x[aatype])
    
    mask = pkl_file['atom_mask'][:, :3]  # (N, 3)
    if include_o:
        mask = np.concatenate([mask, pkl_file['atom_mask'][:, 4:5]], axis=1)  # (N, 4)
    mask = mask[seq_mask]
    
    return coord, seq, mask.astype(bool)

tmalign_patterns = {
    "rmsd": r"RMSD=\s*(\d+\.\d+)",
    "tm_score": r"TM-score=\s*(\d+\.\d+)"
}

def extract_metrics_from_tmalign_output(tmalign_output):
    results = {}
    for line in tmalign_output.strip().split("\n"):
        for key, pattern in tmalign_patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                if key in ["rmsd", "tm_score"]:
                    results[key] = float(value)
                break
    return results

def extract_tm_from_tmalign_output(tmalign_output):
    tm = None
    for line in tmalign_output.strip().split("\n"):
        match = re.search(r"TM-score=\s*(\d+\.\d+)", line)
        if match:
            tm = float(match.group(1))
            break
    return tm

def cal_bb_rmsd(generated_bb_pos, predicted_bb_pos):
    
    generated_bb_pos = generated_bb_pos.reshape(-1, 3)
    predicted_bb_pos = predicted_bb_pos.reshape(-1, 3)
    generated_bb_pos = rigid_transform_3D(generated_bb_pos, predicted_bb_pos)[0]
    di2 = ((generated_bb_pos - predicted_bb_pos) ** 2).sum(axis=-1)
    rmsd = np.sqrt(di2.mean())

    return rmsd

def cal_bb_tm(generated_bb_pos, predicted_bb_pos, seq):

    tm_score = tm_align(generated_bb_pos, predicted_bb_pos, seq, seq).tm_norm_chain1
    
    return tm_score

patterns = {
    "rmsd": r"^RMSD of  the common residues=\s*([\d.]+)",
    "tm_score": r"^TM-score\s*=\s*([\d.]+)",
}

def extract_metrics_from_tmscore_output(tmscore_output):

    results = {}
    for line in tmscore_output.strip().split("\n"):
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                if key in ["structure1_length", "structure2_length", "common_residues"]:
                    results[key] = int(value)
                elif key in ["rmsd", "tm_score", "maxsub_score", "gdt_ts_score", "gdt_ha_score"]:
                    results[key] = float(value)
                break

    return results
