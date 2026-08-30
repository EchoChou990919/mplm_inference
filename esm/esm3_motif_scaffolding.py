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
    
    for pdb, ori_pdb in motif_name_mapping.items():
        print(f'Motif-Scaffolding for {pdb}...')
        
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
                # We'll have to first construct a `GenerationConfig` object that specifies the decoding parameters that we want to use
                # Determine num_steps for sequence: use seq_num_steps if specified (>0), otherwise calculate from sequence length
                if args.seq_num_steps > 0:
                    seq_num_steps = args.seq_num_steps
                else:
                    seq_num_steps = init_prot.sequence.count("_") // args.seq_protlen_steps
                
                sequence_generation_config = GenerationConfig(
                    track="sequence",  # We want ESM3 to generate tokens for the sequence track
                    num_steps=seq_num_steps,
                    temperature=args.seq_temp,
                    strategy=args.seq_unmasking_strategy,
                    temperature_annealing=bool(args.seq_temp_annealing),
                    allow_remask=bool(args.seq_remasking),
                    invalid_ids=seq_invalid_ids,
                )

                sequence_generation = model.generate(init_prot, sequence_generation_config)

                # Determine num_steps for structure: use struct_num_steps if specified (>0), otherwise calculate from sequence length
                if args.struct_num_steps > 0:
                    struct_num_steps = args.struct_num_steps
                else:
                    struct_num_steps = len(sequence_generation) // args.struct_protlen_steps
                
                structure_prediction_config = GenerationConfig(
                    track="structure",  # We want ESM3 to generate tokens for the structure track
                    num_steps=struct_num_steps,
                    temperature=args.struct_temp,
                    strategy=args.struct_unmasking_strategy,
                    temperature_annealing=bool(args.struct_temp_annealing),
                    allow_remask=bool(args.struct_remasking),
                )

                structure_prediction_prompt = ESMProtein(sequence=sequence_generation.sequence)
                structure_prediction = model.generate(
                    structure_prediction_prompt, structure_prediction_config
                )

                # Convert the generated structure to a back into a ProteinChain object
                structure_prediction_chain = structure_prediction.to_protein_chain()

                seq_len = len(sequence_generation.sequence)
                # save the generated sequence
                seq_fasta[f'sample_{idx}_L={seq_len}'] = sequence_generation.sequence
                # save the generated structure as a PDB file
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
                        ),
                        "structure": GenerationTrackConfig(
                            num_steps=num_steps,
                            temperature=args.struct_temp,
                            temperature_annealing=bool(args.struct_temp_annealing),
                            strategy=args.struct_unmasking_strategy,
                            allow_remask=bool(args.struct_remasking),
                        ),
                    },
                    # condition_on_coordinates_only=False
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
                raise NotImplementedError(f"Only seq2struct generation order is currently supported, but got {args.seq_struct_order}")
        
        # save the sequences and start_end_scaffold info
        seq_fasta.write(f"{saveto}/{pdb}.fasta")
        os.makedirs(f"{saveto}/start_end_scaffold", exist_ok=True)
        np.savez(
            f"{saveto}/start_end_scaffold/{pdb}.npz",
            start_idxs_list=start_idxs_list,
            end_idxs_list=end_idxs_list,
            scaffold_length_list=scaffold_length_list
        )
    
    print('Motif-Scaffolding completed!')
    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="esm3-open") # "esm3-open" from Hugging Face
    # or, "esm3-small-2024-08", "esm3-medium-2024-08", "esm3-large-2024-03" from Async Forge Client
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
    parser.add_argument('--seq_remasking', type=int, default=0, help="Whether to allow remasking for sequence generation")
    
    parser.add_argument('--struct_temp', type=float, default=0.7)
    parser.add_argument('--struct_num_steps', type=int, default=-1)
    parser.add_argument('--struct_protlen_steps', type=int, default=2)
    parser.add_argument('--struct_unmasking_strategy', type=str, default="random", choices=["random", "entropy", "stochastic"])
    parser.add_argument('--struct_temp_annealing', type=int, default=1)
    parser.add_argument('--struct_remasking', type=int, default=0, help="Whether to allow remasking for structure generation")

    parser.add_argument('--prot_num', type=int, default=100)
    parser.add_argument('--disable_tqdm', action="store_true", help="Disable progress bar")

    parser.add_argument("--seq_struct_order", type=str, default="seq2struct", 
                        choices=["seq2struct", "cogen"], help="Choose either sequence-to-structure generation or co-generation")

    
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
