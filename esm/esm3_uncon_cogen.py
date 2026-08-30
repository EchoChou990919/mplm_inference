# NOTE: This file was created for this mplm_inference repository

import argparse
import os
from biotite.sequence.io.fasta import FastaFile
from huggingface_hub import login

import esm
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESM3InferenceClient, 
    ESMProtein, 
    GenerationConfig, 
    ESMProteinError, 
    MultiTrackGenerationConfig, 
    GenerationTrackConfig,
)
from esm.tokenization import get_invalid_tokenizer_ids

from utils import seed_everything

def generate(args, model):
    print("Key generation settings:")
    print("============================")
    print(f"Sequence-Structure order: {args.seq_struct_order}")
    print("----------------------------")
    print(f"Sequence num steps: {args.seq_num_steps}")
    print(f"Sequence protlen steps: {args.seq_protlen_steps}")
    print(f"Sequence temperature: {args.seq_temp}")
    print(f"Sequence temperature annealing: {args.seq_temp_annealing}")
    print(f"Sequence unmasking strategy: {args.seq_unmasking_strategy}")
    print(f"Sequence remasking: {args.seq_remasking}")
    print("----------------------------")
    print(f"Structure num steps: {args.struct_num_steps}")
    print(f"Structure protlen steps: {args.struct_protlen_steps}")
    print(f"Structure temperature: {args.struct_temp}")
    print(f"Structure temperature annealing: {args.struct_temp_annealing}")
    print(f"Structure unmasking strategy: {args.struct_unmasking_strategy}")
    print(f"Structure remasking: {args.struct_remasking}")
    print("----------------------------")

    seq_invalid_ids = list(get_invalid_tokenizer_ids(model.tokenizers.sequence))

    output_root = args.output_root
    
    lens = [int(x) for x in args.prot_len.split("-")]
    for prot_len in lens:
        output_dir = os.path.join(output_root, "decoded_pdb", f"length_{prot_len}")
        os.makedirs(output_dir, exist_ok=True)
        fasta_file = FastaFile()
        for i in range(100):
            prompt = "_" * prot_len
            protein = ESMProtein(prompt)

            if args.seq_struct_order == "seq2struct":
                # Determine num_steps for sequence: use seq_num_steps if specified (>0), otherwise calculate from sequence length
                if args.seq_num_steps > 0:
                    seq_num_steps = args.seq_num_steps
                else:
                    seq_num_steps = prot_len // args.seq_protlen_steps
                protein = model.generate(protein, GenerationConfig(
                        track="sequence", 
                        num_steps=seq_num_steps, 
                        temperature=args.seq_temp,
                        temperature_annealing=bool(args.seq_temp_annealing),
                        strategy=args.seq_unmasking_strategy,
                        enable_sequence_resample=bool(args.enable_sequence_resample),
                        resample_ratio=args.resample_ratio,
                        resample_temperature=args.resample_temperature,
                        allow_remask=bool(args.seq_remasking),
                        invalid_ids=seq_invalid_ids,
                    ))
                
                # Determine num_steps for structure: use struct_num_steps if specified (>0), otherwise calculate from sequence length
                if args.struct_num_steps > 0:
                    struct_num_steps = args.struct_num_steps
                else:
                    struct_num_steps = prot_len // args.struct_protlen_steps
                protein = model.generate(protein, GenerationConfig(
                        track="structure", 
                        num_steps=struct_num_steps, 
                        temperature=args.struct_temp,
                        temperature_annealing=bool(args.struct_temp_annealing),
                        strategy=args.struct_unmasking_strategy,
                        allow_remask=bool(args.struct_remasking),
                    ))
            
            elif args.seq_struct_order == "cogen":
                if args.seq_num_steps > 0:
                    num_steps = args.seq_num_steps
                else:
                    num_steps = prot_len // args.seq_protlen_steps
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
                )
                
                protein_tensor = model.encode(protein)
                protein = model.batch_generate([protein_tensor], [multitrack_config])[0]
                # transform ESMProteinTensor back to ESMProtein
                protein = model.decode(protein)
            
            elif args.seq_struct_order == "ss2struct2seq":
                # Determine num_steps for secondary structure: use ss_num_steps if specified (>0), otherwise calculate from sequence length
                if args.ss_num_steps > 0:
                    ss_num_steps = args.ss_num_steps
                else:
                    ss_num_steps = prot_len // args.ss_protlen_steps
                protein = model.generate(protein, GenerationConfig(
                        track="secondary_structure", 
                        num_steps=ss_num_steps, 
                        temperature=args.ss_temp,
                        temperature_annealing=bool(args.ss_temp_annealing),
                        strategy=args.ss_unmasking_strategy,
                    ))
                
                # Determine num_steps for structure: use struct_num_steps if specified (>0), otherwise calculate from sequence length
                if args.struct_num_steps > 0:
                    struct_num_steps = args.struct_num_steps
                else:
                    struct_num_steps = prot_len // args.struct_protlen_steps
                protein = model.generate(protein, GenerationConfig(
                        track="structure", 
                        num_steps=struct_num_steps, 
                        temperature=args.struct_temp,
                        temperature_annealing=bool(args.struct_temp_annealing),
                        strategy=args.struct_unmasking_strategy,
                        allow_remask=bool(args.struct_remasking),
                    ))
                
                # Determine num_steps for sequence: use seq_num_steps if specified (>0), otherwise calculate from sequence length
                if args.seq_num_steps > 0:
                    seq_num_steps = args.seq_num_steps
                else:
                    seq_num_steps = prot_len // args.seq_protlen_steps
                protein = model.generate(protein, GenerationConfig(
                        track="sequence", 
                        num_steps=seq_num_steps, 
                        temperature=args.seq_temp,
                        temperature_annealing=bool(args.seq_temp_annealing),
                        strategy=args.seq_unmasking_strategy,
                        allow_remask=bool(args.seq_remasking),
                        invalid_ids=seq_invalid_ids,
                    ))
            
            else:
                print("Please specify either seq2struct, ss2struct2seq, or cogen for generation.")
                return None

            if isinstance(protein, ESMProteinError):
                print(f"Error in generating sample_{i}, skipping...")
                print(f"Error details: {protein}")
                continue
            elif isinstance(protein, ESMProtein):
                protein.to_pdb(f"{output_dir}/sample_{i}.pdb")
                fasta_file[f"sample_{i}"] = protein.sequence
            else:
                print(f"Unexpected type for protein generated from sample_{i}, skipping...")
                continue
            
        fasta_file.write(os.path.join(output_root, f"length_{prot_len}.fasta"))
    
    return None

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="esm3-open") # "esm3-open" from Hugging Face
    # or, "esm3-small-2024-08", "esm3-medium-2024-08", "esm3-large-2024-03" from Async Forge Client
    parser.add_argument("--forge_token", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--prot_len", type=str, default="100-200-300-400-500")
    parser.add_argument("--output_root", type=str, default="esm/generation-results/debug")
    
    parser.add_argument("--seq_num_steps", type=int, default=-1)
    parser.add_argument("--seq_protlen_steps", type=int, default=8)
    parser.add_argument("--seq_temp", type=float, default=1.0)
    parser.add_argument("--seq_temp_annealing", type=int, default=1)
    parser.add_argument("--seq_unmasking_strategy", type=str, default="random", choices=["random", "entropy", "stochastic"])
    parser.add_argument("--seq_remasking", type=int, default=0)

    parser.add_argument("--struct_num_steps", type=int, default=1)
    parser.add_argument("--struct_protlen_steps", type=int, default=1)
    parser.add_argument("--struct_temp", type=float, default=0.0)
    parser.add_argument("--struct_temp_annealing", type=int, default=1)
    parser.add_argument("--struct_unmasking_strategy", type=str, default="random", choices=["random", "entropy", "stochastic"])
    parser.add_argument("--struct_remasking", type=int, default=0)

    parser.add_argument("--ss_num_steps", type=int, default=-1)
    parser.add_argument("--ss_protlen_steps", type=int, default=1)
    parser.add_argument("--ss_temp", type=float, default=1.0)
    parser.add_argument("--ss_temp_annealing", type=int, default=1)
    parser.add_argument("--ss_unmasking_strategy", type=str, default="random", choices=["random", "entropy", "stochastic"])

    parser.add_argument("--enable_sequence_resample", type=int, default=0)
    parser.add_argument("--resample_ratio", type=float, default=0.25)
    parser.add_argument("--resample_temperature", type=float, default=None)

    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bar")

    # parser.add_argument("--seq_struct_order", type=str, required=True, 
    #                     choices=["seq2struct", "ss2struct2seq", "cogen"], help="Choose either sequence-to-structure generation or co-generation")
    parser.add_argument("--seq_struct_order", type=str, default="seq2struct", 
                        choices=["seq2struct", "ss2struct2seq", "cogen"], help="Choose either sequence-to-structure generation or co-generation")

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

if __name__ == "__main__":
    main()