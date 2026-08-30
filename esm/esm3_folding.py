# NOTE: This file was created for this mplm_inference repository

import argparse
import os
from biotite.sequence.io.fasta import FastaFile
from huggingface_hub import login

import esm
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

from utils import seed_everything

def folding(args, model):
    fasta_path = args.input_fasta_path
    output_dir = args.output_dir
    
    print(f'Folding for {fasta_path}...')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "decoded_pdb"), exist_ok=True)

    fasta_file = FastaFile.read(fasta_path)
    all_sequences = fasta_file.items()
    
    for pdb_name, sequence in all_sequences:
        prompt = sequence
        protein = ESMProtein(prompt)
        
        # Determine num_steps for structure: use struct_num_steps if specified (>0), otherwise calculate from sequence length
        if args.struct_num_steps > 0:
            struct_num_steps = args.struct_num_steps
        else:
            struct_num_steps = len(sequence) // args.struct_protlen_steps
        
        protein = model.generate(protein, GenerationConfig(
            track="structure", 
            strategy=args.unmasking_strategy,
            num_steps=struct_num_steps, 
            temperature=args.struct_temp,
            temperature_annealing=bool(args.temp_annealing),
            allow_remask=bool(args.remasking))
        )
        protein.to_pdb(f"{output_dir}/decoded_pdb/{pdb_name}.pdb")
    
    return None

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="esm3-open") # "esm3-open" from Hugging Face
    # or, "esm3-small-2024-08", "esm3-medium-2024-08", "esm3-large-2024-03" from Async Forge Client
    parser.add_argument("--forge_token", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--input_fasta_path", type=str, default="", help="Path to input FASTA file")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for results")

    parser.add_argument("--struct_num_steps", type=int, default=8)
    parser.add_argument("--struct_protlen_steps", type=int, default=-1)
    parser.add_argument("--struct_temp", type=float, default=0.7)
    parser.add_argument("--unmasking_strategy", type=str, default="entropy")
    parser.add_argument("--remasking", type=int, default=1, help="Whether to allow remasking of already-decoded positions (like DPLM2's reparameterized decoding)")
    parser.add_argument("--temp_annealing", type=int, default=0)
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bar")
    
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

    folding(args, model)

if __name__ == "__main__":
    main()