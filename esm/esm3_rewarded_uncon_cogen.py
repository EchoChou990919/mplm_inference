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
    ESMProteinError,
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
    seq_invalid_ids = list(get_invalid_tokenizer_ids(model.tokenizers.sequence))

    print("Key generation settings:")
    print("============================")
    print(f"CFG: seq_cfg_scale={args.seq_cfg_scale}, struct_cfg_scale={args.struct_cfg_scale}, cfg_mode={args.cfg_mode}")
    print(f"Beam search: beam_width={args.beam_width}, branching_factor={args.branching_factor}, scoring_interval={args.scoring_interval}")
    print(f"Reward: alpha={args.reward_alpha}, beta={args.reward_beta}, seq_reward_type={args.seq_reward_type}, reward_mode={args.reward_mode}")
    print("----------------------------")
    print(f"Sequence: temp={args.seq_temp}, protlen_steps={args.seq_protlen_steps}, num_steps={args.seq_num_steps}, "
          f"strategy={args.seq_unmasking_strategy}, annealing={args.seq_temp_annealing}, remasking={args.seq_remasking}")
    print(f"Structure: temp={args.struct_temp}, protlen_steps={args.struct_protlen_steps}, num_steps={args.struct_num_steps}, "
          f"strategy={args.struct_unmasking_strategy}, annealing={args.struct_temp_annealing}, remasking={args.struct_remasking}")
    print("============================")

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

    output_root = args.output_root

    lens = [int(x) for x in args.prot_len.split("-")]
    for prot_len in lens:
        output_dir = os.path.join(output_root, "decoded_pdb", f"length_{prot_len}")
        os.makedirs(output_dir, exist_ok=True)
        fasta_file = FastaFile()
        for i in range(100):
            prompt = "_" * prot_len
            protein = ESMProtein(prompt)

            # Determine num_steps
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

            # Determine K: adaptive (num_scoring_rounds) > fixed (scoring_interval)
            if args.num_scoring_rounds > 0:
                K = max(1, num_steps // args.num_scoring_rounds)
            else:
                K = args.scoring_interval
                if K <= 0:
                    K = num_steps  # K=T means Best-of-N

            result = beam_search.search(
                protein,
                multitrack_config,
                beam_width=args.beam_width,
                branching_factor=args.branching_factor,
                scoring_interval=K,
                verbose=not args.disable_tqdm,
                selection_mode=args.selection_mode,
                struct_reward_threshold=args.struct_reward_threshold,
                seq_reward_threshold=args.seq_reward_threshold,
            )

            if isinstance(result, ESMProteinError):
                print(f"Error in generating sample_{i}, skipping...")
                print(f"Error details: {result}")
                continue
            elif isinstance(result, ESMProtein):
                result.to_pdb(f"{output_dir}/sample_{i}.pdb")
                fasta_file[f"sample_{i}"] = result.sequence
            else:
                print(f"Unexpected type for protein generated from sample_{i}, skipping...")
                continue

        fasta_file.write(os.path.join(output_root, f"length_{prot_len}.fasta"))

    print('Rewarded Unconditional Cogeneration completed!')


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

    parser.add_argument("--struct_num_steps", type=int, default=-1)
    parser.add_argument("--struct_protlen_steps", type=int, default=1)
    parser.add_argument("--struct_temp", type=float, default=0.7)
    parser.add_argument("--struct_temp_annealing", type=int, default=1)
    parser.add_argument("--struct_unmasking_strategy", type=str, default="random", choices=["random", "entropy", "stochastic"])
    parser.add_argument("--struct_remasking", type=int, default=0)

    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bar")

    # CFG arguments
    parser.add_argument("--seq_cfg_scale", type=float, default=0.0, help="CFG guidance scale for sequence track")
    parser.add_argument("--struct_cfg_scale", type=float, default=0.0, help="CFG guidance scale for structure track")
    parser.add_argument("--cfg_mode", type=str, default="cross_modal",
                        choices=["drop_condition", "full_mask", "cross_modal"],
                        help="CFG mode for unconditional co-generation")

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


if __name__ == "__main__":
    main()
