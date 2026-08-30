# NOTE: This file was created for this mplm_inference repository

import argparse
import os
from biotite.sequence.io.fasta import FastaFile
from huggingface_hub import login

import esm
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.sdk.experimental import (
    GeneralizedBeamSearch,
    StructureReward,
)

from utils import seed_everything


def folding(args, model):
    fasta_path = args.input_fasta_path
    output_dir = args.output_dir

    print(f'Rewarded Folding for {fasta_path}...')
    print(f'cfg_scale={args.cfg_scale}, cfg_mode={args.cfg_mode}')
    print(f'beam_width={args.beam_width}, branching_factor={args.branching_factor}, '
          f'scoring_interval={args.scoring_interval}, reward_mode={args.reward_mode}')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "decoded_pdb"), exist_ok=True)

    fasta_file = FastaFile.read(fasta_path)
    all_sequences = fasta_file.items()

    # Build reward function
    reward_fn = StructureReward(mode=args.reward_mode)

    # Build beam search engine
    beam_search = GeneralizedBeamSearch(
        client=model,
        reward_fn=reward_fn,
        tokenizers=model.tokenizers,
    )

    for pdb_name, sequence in all_sequences:
        prompt = sequence
        protein = ESMProtein(prompt)

        # Determine num_steps for structure
        if args.struct_num_steps > 0:
            struct_num_steps = args.struct_num_steps
        else:
            struct_num_steps = len(sequence) // args.struct_protlen_steps

        config = GenerationConfig(
            track="structure",
            strategy=args.unmasking_strategy,
            num_steps=struct_num_steps,
            temperature=args.struct_temp,
            temperature_annealing=bool(args.temp_annealing),
            allow_remask=bool(args.remasking),
            cfg_scale=args.cfg_scale,
            cfg_mode=args.cfg_mode,
        )

        # Determine K: adaptive (num_scoring_rounds) > fixed (scoring_interval)
        if args.num_scoring_rounds > 0:
            K = max(1, struct_num_steps // args.num_scoring_rounds)
        else:
            K = args.scoring_interval
            if K <= 0:
                K = struct_num_steps  # K=T means Best-of-N

        protein = beam_search.search(
            protein,
            config,
            beam_width=args.beam_width,
            branching_factor=args.branching_factor,
            scoring_interval=K,
            verbose=not args.disable_tqdm,
        )
        protein.to_pdb(f"{output_dir}/decoded_pdb/{pdb_name}.pdb")

    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="esm3-open")
    parser.add_argument("--forge_token", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--input_fasta_path", type=str, default="/path/to/mplm_sampling_neurips/data/cameo2022/aatype.fasta", help="Path to input FASTA file")
    parser.add_argument("--output_dir", type=str, default="/path/to/mplm_sampling_neurips/esm/generation-results/esm3-open/rewarded_folding", help="Output directory for results")

    parser.add_argument("--struct_num_steps", type=int, default=8)
    parser.add_argument("--struct_protlen_steps", type=int, default=-1)
    parser.add_argument("--struct_temp", type=float, default=0.7)
    parser.add_argument("--unmasking_strategy", type=str, default="entropy")
    parser.add_argument("--remasking", type=int, default=1, help="Whether to allow remasking of already-decoded positions (like DPLM2's reparameterized decoding)")
    parser.add_argument("--temp_annealing", type=int, default=0)
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bar")

    # CFG arguments
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG guidance scale. 0.0 means no guidance.")
    parser.add_argument("--cfg_mode", type=str, default="drop_condition", choices=["drop_condition", "full_mask"], help="CFG mode: drop_condition masks non-target tracks; full_mask masks all tracks.")

    # Beam search arguments
    parser.add_argument("--beam_width", type=int, default=1, help="N: number of beams maintained in parallel")
    parser.add_argument("--branching_factor", type=int, default=4, help="L: candidates generated per beam at scoring steps")
    parser.add_argument("--scoring_interval", type=int, default=1, help="K: score every K steps. K<=0 means K=T (Best-of-N)")
    parser.add_argument("--num_scoring_rounds", type=int, default=0, help="Adaptive K: when >0, K=max(1, T//num_scoring_rounds). Overrides --scoring_interval")
    parser.add_argument("--reward_mode", type=str, default="ptm", choices=["ptm", "plddt"], help="Structure reward mode")

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
