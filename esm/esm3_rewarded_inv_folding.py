# NOTE: This file was created for this mplm_inference repository

import argparse
import os
import pickle
from biotite.sequence.io.fasta import FastaFile
from huggingface_hub import login

import esm
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.tokenization import get_invalid_tokenizer_ids
from esm.sdk.experimental import (
    GeneralizedBeamSearch,
    SequenceFoldabilityReward,
    SequenceLogprobReward,
)

from utils import seed_everything


def inv_folding(args, model):
    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f'Rewarded Inverse Folding for {input_dir}...')
    print(f'cfg_scale={args.cfg_scale}, cfg_mode={args.cfg_mode}')
    print(f'beam_width={args.beam_width}, branching_factor={args.branching_factor}, '
          f'scoring_interval={args.scoring_interval}, reward_type={args.reward_type}')

    seq_invalid_ids = list(get_invalid_tokenizer_ids(model.tokenizers.sequence))

    os.makedirs(output_dir, exist_ok=True)

    seq_fasta = FastaFile.read(os.path.join(input_dir, "aatype.fasta"))
    pdb_names = list(seq_fasta.keys())

    # Build reward function
    if args.reward_type == "foldability":
        reward_fn = SequenceFoldabilityReward(mode=args.reward_mode)
    else:
        reward_fn = SequenceLogprobReward(use_full_context_logprob=bool(args.full_context_logprob))

    # Build beam search engine
    beam_search = GeneralizedBeamSearch(
        client=model,
        reward_fn=reward_fn,
        tokenizers=model.tokenizers,
    )

    inv_folding_seqs = FastaFile()
    for pdb_name in pdb_names:

        file_path = os.path.join(input_dir, "preprocessed", f"{pdb_name}.pkl")
        with open(file_path, "rb") as f:
            pkl_data = pickle.load(f)

        aatype = pkl_data['aatype']
        seq_mask = (aatype < 20)
        seq_mask_idx = [i for i, m in enumerate(seq_mask) if m]
        seq_mask[seq_mask_idx[0]: seq_mask_idx[-1] + 1] = True  # make the mask continuous

        coords = pkl_data['atom_positions']  # (N, 37, 3)
        coords = coords[seq_mask]

        protein = ESMProtein(coordinates=coords)

        # Determine num_steps for sequence
        if args.seq_num_steps > 0:
            seq_num_steps = args.seq_num_steps
        else:
            seq_num_steps = len(coords) // args.seq_protlen_steps

        config = GenerationConfig(
            track="sequence",
            num_steps=seq_num_steps,
            temperature=args.seq_temp,
            strategy=args.unmasking_strategy,
            temperature_annealing=bool(args.temp_annealing),
            allow_remask=bool(args.remasking),
            cfg_scale=args.cfg_scale,
            cfg_mode=args.cfg_mode,
            invalid_ids=seq_invalid_ids,
        )

        # Determine K: adaptive (num_scoring_rounds) > fixed (scoring_interval)
        if args.num_scoring_rounds > 0:
            K = max(1, seq_num_steps // args.num_scoring_rounds)
        else:
            K = args.scoring_interval
            if K <= 0:
                K = seq_num_steps  # K=T means Best-of-N

        protein = beam_search.search(
            protein,
            config,
            beam_width=args.beam_width,
            branching_factor=args.branching_factor,
            scoring_interval=K,
            verbose=not args.disable_tqdm,
        )

        inv_folding_seqs[pdb_name] = protein.sequence

    inv_folding_seqs.write(os.path.join(output_dir, "aatype.fasta"))

    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="esm3-open")
    parser.add_argument("--forge_token", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--input_dir", type=str, default="", help="Path to input dataset directory")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for results")

    parser.add_argument("--seq_num_steps", type=int, default=-1)
    parser.add_argument("--seq_protlen_steps", type=int, default=8)
    parser.add_argument("--seq_temp", type=float, default=1.0)
    parser.add_argument("--unmasking_strategy", type=str, default="random")
    parser.add_argument("--temp_annealing", type=int, default=1)
    parser.add_argument("--remasking", type=int, default=1, help="Whether to allow remasking of already-decoded positions (like DPLM2's reparameterized decoding)")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bar")

    # CFG arguments
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG guidance scale. 0.0 means no guidance.")
    parser.add_argument("--cfg_mode", type=str, default="drop_condition", choices=["drop_condition", "full_mask"], help="CFG mode")

    # Beam search arguments
    parser.add_argument("--beam_width", type=int, default=1, help="N: number of beams maintained in parallel")
    parser.add_argument("--branching_factor", type=int, default=4, help="L: candidates generated per beam at scoring steps")
    parser.add_argument("--scoring_interval", type=int, default=1, help="K: score every K steps. K<=0 means K=T (Best-of-N)")
    parser.add_argument("--num_scoring_rounds", type=int, default=0, help="Adaptive K: when >0, K=max(1, T//num_scoring_rounds). Overrides --scoring_interval")
    parser.add_argument("--reward_type", type=str, default="foldability", choices=["foldability", "logprob"], help="Reward type: foldability (default, stronger signal) or logprob (sequence log-probability)")
    parser.add_argument("--reward_mode", type=str, default="ptm", choices=["ptm", "plddt", "nan"], help="Reward mode for foldability reward: ptm or plddt; nan is allowed only for logprob mode")
    parser.add_argument("--full_context_logprob", type=int, default=0, help="Use full-context logprob (Option B) instead of shared forward (Option A). Only used when reward_type=logprob")

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

    inv_folding(args, model)


if __name__ == "__main__":
    main()
