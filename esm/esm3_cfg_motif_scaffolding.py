# NOTE: This file was created for this mplm_inference repository

import argparse

import numpy as np

import torch
import os

from scaffolding_utils import motif_name_mapping, get_initial_scaffolding_batches

from biotite.sequence.io.fasta import FastaFile

from huggingface_hub import login

import esm
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESM3InferenceClient, 
    ESMProtein, 
    MultiTrackGenerationConfig,
    GenerationTrackConfig,
    GenerationConfig
)
from esm.tokenization import get_invalid_tokenizer_ids

from utils import seed_everything

def generate(args, model):
    saveto = args.saveto
    seq_invalid_ids = list(get_invalid_tokenizer_ids(model.tokenizers.sequence))

    if args.seq_struct_order == "seq2struct" and args.cfg_mode in ("cross_modal", "motif_anchored_cross_modal", "drop_motif_cross_modal"):
        raise ValueError(
            f"cfg_mode='{args.cfg_mode}' is not supported with seq2struct generation order. "
            f"{args.cfg_mode} requires simultaneous multi-track generation (use --seq_struct_order cogen). "
            "For seq2struct scaffolding, use 'drop_motif', 'drop_condition', or 'full_mask'."
        )

    for pdb, ori_pdb in motif_name_mapping.items():
        print(f'CFG Motif-Scaffolding for {pdb}...')
        print(f'seq_cfg_scale={args.seq_cfg_scale}, struct_cfg_scale={args.struct_cfg_scale}, cfg_mode={args.cfg_mode}')
        
        pdb_file = f'{args.data_dir}/{ori_pdb}_reference.pdb'
        (
            init_prots, start_idxs_list, 
            end_idxs_list, scaffold_length_list
        ) = get_initial_scaffolding_batches(
            pdb_file, pdb, ori_pdb, args.prot_num, args.device
        )
        
        seq_fasta = FastaFile()
        for idx, init_prot in enumerate(init_prots):
            
            if args.seq_struct_order == "seq2struct":
                # Determine num_steps for sequence
                if args.seq_num_steps > 0:
                    seq_num_steps = args.seq_num_steps
                else:
                    seq_num_steps = init_prot.sequence.count("_") // args.seq_protlen_steps
                
                sequence_generation_config = GenerationConfig(
                    track="sequence",
                    num_steps=seq_num_steps,
                    temperature=args.seq_temp,
                    strategy=args.seq_unmasking_strategy,
                    temperature_annealing=bool(args.seq_temp_annealing),
                    allow_remask=bool(args.seq_remasking),
                    invalid_ids=seq_invalid_ids,
                    cfg_scale=args.seq_cfg_scale,
                    cfg_mode=args.cfg_mode,
                )

                sequence_generation = model.generate(init_prot, sequence_generation_config)

                # Determine num_steps for structure
                if args.struct_num_steps > 0:
                    struct_num_steps = args.struct_num_steps
                else:
                    struct_num_steps = len(sequence_generation) // args.struct_protlen_steps
                
                structure_prediction_config = GenerationConfig(
                    track="structure",
                    num_steps=struct_num_steps,
                    temperature=args.struct_temp,
                    strategy=args.struct_unmasking_strategy,
                    temperature_annealing=bool(args.struct_temp_annealing),
                    allow_remask=bool(args.struct_remasking),
                    cfg_scale=args.struct_cfg_scale,
                    cfg_mode=args.cfg_mode,
                )

                structure_prediction_prompt = ESMProtein(sequence=sequence_generation.sequence)
                structure_prediction = model.generate(
                    structure_prediction_prompt, structure_prediction_config
                )

                structure_prediction_chain = structure_prediction.to_protein_chain()

                seq_len = len(sequence_generation.sequence)
                seq_fasta[f'sample_{idx}_L={seq_len}'] = sequence_generation.sequence
                save_path = os.path.join(saveto, 'decoded_pdb', pdb)
                os.makedirs(save_path, exist_ok=True)
                with open(f'{save_path}/struct_{idx}.pdb', 'w') as f:
                    f.write(structure_prediction_chain.to_pdb_string())
            
            elif args.seq_struct_order == "cogen":
                if args.seq_num_steps > 0:
                    num_steps = args.seq_num_steps
                else:
                    num_steps = init_prot.sequence.count("_") // args.seq_protlen_steps
                
                multitrack_config = MultiTrackGenerationConfig(
                    tracks={
                        "sequence": GenerationTrackConfig(
                            num_steps=num_steps,
                            temperature=args.seq_temp,
                            temperature_annealing=bool(args.seq_temp_annealing),
                            strategy=args.seq_unmasking_strategy,
                            allow_remask=bool(args.seq_remasking),
                            invalid_ids=seq_invalid_ids,
                            cfg_scale=args.seq_cfg_scale,
                            cfg_mode=args.cfg_mode,
                        ),
                        "structure": GenerationTrackConfig(
                            num_steps=num_steps,
                            temperature=args.struct_temp,
                            temperature_annealing=bool(args.struct_temp_annealing),
                            strategy=args.struct_unmasking_strategy,
                            allow_remask=bool(args.struct_remasking),
                            cfg_scale=args.struct_cfg_scale,
                            cfg_mode=args.cfg_mode,
                        ),
                    },
                )

                init_prot_tensor = model.encode(init_prot)
                protein = model.batch_generate([init_prot_tensor], [multitrack_config])[0]
                protein = model.decode(protein)

                seq_len = len(protein.sequence)
                seq_fasta[f'sample_{idx}_L={seq_len}'] = protein.sequence
                save_path = os.path.join(saveto, 'decoded_pdb', pdb)
                os.makedirs(save_path, exist_ok=True)
                protein.to_pdb(f"{save_path}/struct_{idx}.pdb")
            else:
                raise NotImplementedError(f"Only seq2struct and cogen are supported, but got {args.seq_struct_order}")
        
        seq_fasta.write(f"{saveto}/{pdb}.fasta")
        os.makedirs(f"{saveto}/start_end_scaffold", exist_ok=True)
        np.savez(
            f"{saveto}/start_end_scaffold/{pdb}.npz",
            start_idxs_list=start_idxs_list,
            end_idxs_list=end_idxs_list,
            scaffold_length_list=scaffold_length_list
        )
    
    print('CFG Motif-Scaffolding completed!')
    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="esm3-open")
    parser.add_argument("--forge_token", type=str, default="")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda:0")
    
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--saveto', type=str, default='')

    parser.add_argument('--seq_temp', type=float, default=0.5)
    parser.add_argument('--seq_num_steps', type=int, default=-1)
    parser.add_argument('--seq_protlen_steps', type=int, default=8)
    parser.add_argument('--seq_unmasking_strategy', type=str, default="random", choices=["random", "entropy", "stochastic"])
    parser.add_argument('--seq_temp_annealing', type=int, default=1)
    parser.add_argument('--seq_remasking', type=int, default=0)
    
    parser.add_argument('--struct_temp', type=float, default=0.7)
    parser.add_argument('--struct_num_steps', type=int, default=-1)
    parser.add_argument('--struct_protlen_steps', type=int, default=2)
    parser.add_argument('--struct_unmasking_strategy', type=str, default="random", choices=["random", "entropy", "stochastic"])
    parser.add_argument('--struct_temp_annealing', type=int, default=1)
    parser.add_argument('--struct_remasking', type=int, default=0)

    parser.add_argument('--prot_num', type=int, default=100)
    parser.add_argument('--disable_tqdm', action="store_true")

    parser.add_argument("--seq_struct_order", type=str, default="seq2struct", 
                        choices=["seq2struct", "cogen"])

    # CFG arguments
    parser.add_argument("--seq_cfg_scale", type=float, default=0.0, help="CFG guidance scale for sequence track")
    parser.add_argument("--struct_cfg_scale", type=float, default=0.0, help="CFG guidance scale for structure track")
    parser.add_argument("--cfg_mode", type=str, default="drop_motif", 
                        choices=["drop_condition", "drop_motif", "full_mask", "cross_modal", "motif_anchored_cross_modal", "drop_motif_cross_modal"],
                        help="CFG mode for scaffolding")

    args = parser.parse_args()

    if args.disable_tqdm:
        os.environ["DISABLE_ITERATIVE_SAMPLING_TQDM"] = "1"

    seed_everything(seed=args.seed)

    if args.model_name == "esm3-open":
        model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to(args.device)
    elif args.model_name in ["esm3-small-2024-08", "esm3-medium-2024-08", "esm3-large-2024-03"]:
        model: ESM3InferenceClient = esm.sdk.client(args.model_name, token=args.forge_token)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    generate(args, model)


if __name__ == '__main__':
    main()
