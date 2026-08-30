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
    # Validate cfg_mode compatibility
    if args.seq_struct_order == "seq2struct" and args.cfg_mode == "cross_modal":
        raise ValueError(
            "cfg_mode='cross_modal' is not supported with seq2struct generation order. "
            "cross_modal requires simultaneous multi-track generation (use --seq_struct_order cogen). "
            "For seq2struct, use 'drop_condition' or 'full_mask'."
        )

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
                # Determine num_steps for sequence
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
                        cfg_scale=args.seq_cfg_scale,
                        cfg_mode=args.cfg_mode,
                    ))
                
                # Determine num_steps for structure
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
                        cfg_scale=args.struct_cfg_scale,
                        cfg_mode=args.cfg_mode,
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
                
                protein_tensor = model.encode(protein)
                protein = model.batch_generate([protein_tensor], [multitrack_config])[0]
                protein = model.decode(protein)
            
            else:
                print("Please specify either seq2struct or cogen for generation.")
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

    parser.add_argument("--model_name", type=str, default="esm3-open")
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

    parser.add_argument("--enable_sequence_resample", type=int, default=0)
    parser.add_argument("--resample_ratio", type=float, default=0.25)
    parser.add_argument("--resample_temperature", type=float, default=None)

    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bar")

    parser.add_argument("--seq_struct_order", type=str, default="cogen", 
                        choices=["seq2struct", "cogen"], help="Choose either sequence-to-structure generation or co-generation")

    # CFG arguments
    parser.add_argument("--seq_cfg_scale", type=float, default=0.0, help="CFG guidance scale for sequence track")
    parser.add_argument("--struct_cfg_scale", type=float, default=0.0, help="CFG guidance scale for structure track")
    parser.add_argument("--cfg_mode", type=str, default="cross_modal", 
                        choices=["drop_condition", "full_mask", "cross_modal"],
                        help="CFG mode for unconditional co-generation")

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
