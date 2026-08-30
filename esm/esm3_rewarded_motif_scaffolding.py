# NOTE: This file was created for this mplm_inference repository

import argparse
import os

import numpy as np
import torch

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
)
from esm.tokenization import get_invalid_tokenizer_ids
from esm.sdk.experimental import (
    CombinedReward,
    GeneralizedBeamSearch,
    SequenceFoldabilityReward,
    SequenceLogprobReward,
    StructureReward,
)

from utils import seed_everything


def generate(args, model):
    saveto = args.saveto
    seq_invalid_ids = list(get_invalid_tokenizer_ids(model.tokenizers.sequence))

    # Build reward function
    struct_reward = StructureReward(mode="ptm")
    if args.seq_reward_type == "foldability":
        seq_reward = SequenceFoldabilityReward(mode=args.reward_mode)
    else:
        seq_reward = SequenceLogprobReward(use_full_context_logprob=False)
    reward_fn = CombinedReward(
        rewards=[struct_reward, seq_reward],
        weights=[args.reward_alpha, args.reward_beta],
    )

    # Build beam search engine
    beam_search = GeneralizedBeamSearch(
        client=model,
        reward_fn=reward_fn,
        tokenizers=model.tokenizers,
    )

    for pdb, ori_pdb in motif_name_mapping.items():
        print(f'Rewarded Motif-Scaffolding for {pdb}...')
        print(f'seq_cfg_scale={args.seq_cfg_scale}, struct_cfg_scale={args.struct_cfg_scale}, cfg_mode={args.cfg_mode}')
        print(f'beam_width={args.beam_width}, branching_factor={args.branching_factor}, '
              f'scoring_interval={args.scoring_interval}, seq_reward_type={args.seq_reward_type}')
        print(f'reward_alpha={args.reward_alpha}, reward_beta={args.reward_beta}')

        pdb_file = f'{args.data_dir}/{ori_pdb}_reference.pdb'
        (
            init_prots, start_idxs_list,
            end_idxs_list, scaffold_length_list
        ) = get_initial_scaffolding_batches(
            pdb_file, pdb, ori_pdb, args.prot_num, args.device
        )

        seq_fasta = FastaFile()
        for idx, init_prot in enumerate(init_prots):
            # Determine num_steps
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

            # Determine K: adaptive (num_scoring_rounds) > fixed (scoring_interval)
            if args.num_scoring_rounds > 0:
                K = max(1, num_steps // args.num_scoring_rounds)
            else:
                K = args.scoring_interval
                if K <= 0:
                    K = num_steps  # K=T means Best-of-N

            # Encode init_prot to get input_tokens (needed for drop_motif* CFG modes)
            init_prot_tensor = model.encode(init_prot)

            protein = beam_search.search(
                init_prot,
                multitrack_config,
                beam_width=args.beam_width,
                branching_factor=args.branching_factor,
                scoring_interval=K,
                input_tokens=init_prot_tensor,
                verbose=not args.disable_tqdm,
                selection_mode=args.selection_mode,
                struct_reward_threshold=args.struct_reward_threshold,
                seq_reward_threshold=args.seq_reward_threshold,
            )

            seq_len = len(protein.sequence)
            seq_fasta[f'sample_{idx}_L={seq_len}'] = protein.sequence
            save_path = os.path.join(saveto, 'decoded_pdb', pdb)
            os.makedirs(save_path, exist_ok=True)
            protein.to_pdb(f"{save_path}/struct_{idx}.pdb")

        seq_fasta.write(f"{saveto}/{pdb}.fasta")
        os.makedirs(f"{saveto}/start_end_scaffold", exist_ok=True)
        np.savez(
            f"{saveto}/start_end_scaffold/{pdb}.npz",
            start_idxs_list=start_idxs_list,
            end_idxs_list=end_idxs_list,
            scaffold_length_list=scaffold_length_list
        )

    print('Rewarded Motif-Scaffolding completed!')


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

    # CFG arguments
    parser.add_argument("--seq_cfg_scale", type=float, default=0.0, help="CFG guidance scale for sequence track")
    parser.add_argument("--struct_cfg_scale", type=float, default=0.0, help="CFG guidance scale for structure track")
    parser.add_argument("--cfg_mode", type=str, default="drop_motif",
                        choices=["drop_condition", "drop_motif", "full_mask", "cross_modal", "motif_anchored_cross_modal", "drop_motif_cross_modal"],
                        help="CFG mode for scaffolding")

    # Beam search arguments
    parser.add_argument("--beam_width", type=int, default=1, help="N: number of beams maintained in parallel")
    parser.add_argument("--branching_factor", type=int, default=4, help="L: candidates generated per beam at scoring steps")
    parser.add_argument("--scoring_interval", type=int, default=1, help="K: score every K steps. K<=0 means K=T (Best-of-N)")
    parser.add_argument("--num_scoring_rounds", type=int, default=0, help="Adaptive K: when >0, K=max(1, T//num_scoring_rounds). Overrides --scoring_interval")

    # Reward arguments
    parser.add_argument("--reward_alpha", type=float, default=1.0, help="Weight for StructureReward(pTM)")
    parser.add_argument("--reward_beta", type=float, default=1.0, help="Weight for SequenceFoldabilityReward/SequenceLogprobReward")
    parser.add_argument("--seq_reward_type", type=str, default="foldability", choices=["foldability", "logprob"],
                        help="Sequence-side reward: foldability (default, stronger) or logprob (ablation)")
    parser.add_argument("--reward_mode", type=str, default="ptm", choices=["ptm", "plddt"],
                        help="Reward mode for foldability reward: ptm or plddt")

    # Final selection arguments
    parser.add_argument("--selection_mode", type=str, default="best", choices=["best", "threshold_random"],
                        help="Final selection: best (argmax) or threshold_random (random among beams passing thresholds)")
    parser.add_argument("--struct_reward_threshold", type=float, default=0.5,
                        help="Min structure reward for threshold_random selection")
    parser.add_argument("--seq_reward_threshold", type=float, default=0.5,
                        help="Min sequence reward for threshold_random selection")

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
