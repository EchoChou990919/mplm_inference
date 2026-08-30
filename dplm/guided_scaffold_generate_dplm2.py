# NOTE: This file was created for this mplm_inference repository

import argparse
import os
from pprint import pprint

import biotite.sequence.io.fasta as fasta
import numpy as np
import pandas as pd
import torch
from peft.peft_model import PeftModel
# ======
# NOTE: add bit_model
from byprot.models.dplm2 import DPLM2Bit
from byprot.models.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)
# ======
from byprot.utils.scaffold_utils import *
from generate_dplm2 import save_fasta

from pytorch_lightning import seed_everything

@torch.no_grad()
def generate(args, saveto):
    # ======
    # NOTE: add bit_model
    if args.bit_model:
        model = DPLM2Bit.from_pretrained(args.model_name)
    else:
        model = DPLM2.from_pretrained(args.model_name)
    # ======
    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    # Read motif fasta file
    with open(args.motif_aa, "r") as f:
        fasta_file = fasta.FastaFile.read(f)
        motif_aa_seq = dict(fasta_file.items())
    with open(args.motif_struct, "r") as f:
        fasta_file = fasta.FastaFile.read(f)
        motif_struct_seq = dict(fasta_file.items())

    for ori_pdb_name, pdb_name in motif_name_mapping.items():
        struct_seq = motif_struct_seq[pdb_name]
        aa_seq = motif_aa_seq[pdb_name]
        (
            batches,
            start_idxs_list,
            end_idxs_list,
            scaffold_lengths_list,
        ) = get_initial_dplm2(
            args,
            list(aa_seq),
            struct_seq.split(","),
            tokenizer,
            pdb_name,
            ori_pdb_name,
            device,
        )

        # ======
        # NOTE: adjust max_iter based on batch size and sequence length
        start_idx = 0
        for batch in batches:
            seq_len = (batch["input_ids"].shape[1] - 2) // 2 # exclude start and end tokens
            # ======
            # NOTE: adjust max_iter based on batch size and sequence length
            if args.batch_size < 0:
                max_iter = int(seq_len // args.maxiter_seqlen_ratio)
            else:
                max_iter = args.max_iter
            # ======
            with torch.cuda.amp.autocast():
                output_tokens = model.generate(
                    input_tokens=batch["input_ids"],
                    max_iter=max_iter,
                    unmasking_strategy=args.unmasking_strategy,
                    sampling_strategy=args.sampling_strategy,
                    partial_masks=batch["partial_mask"],
                    # ======
                    # NOTE: CFG support — split cfg_scale
                    seq_cfg_scale=args.seq_cfg_scale,
                    struct_cfg_scale=args.struct_cfg_scale,
                    cfg_mode=args.cfg_mode,
                    # ======
                )["output_tokens"]
            
            print("final:")
            pprint(
                [
                    ",".join(seq.split(" "))
                    for seq in tokenizer.batch_decode(
                        output_tokens, skip_special_tokens=False
                    )
                ]
            )

            # save output
            scaffold_fasta_path = os.path.join(saveto, "scaffold_fasta")
            os.makedirs(scaffold_fasta_path, exist_ok=True)
            scaffold_info_path = os.path.join(saveto, "start_end_scaffold")
            os.makedirs(scaffold_info_path, exist_ok=True)

            # save scaffold fasta
            save_results(
                output_tokens=output_tokens,
                save_dir=os.path.join(scaffold_fasta_path, ori_pdb_name),
                tokenizer=tokenizer,
                struct_tokenizer=model.struct_tokenizer,
                save_pdb=True,
                continue_write=True,
                start_idx=start_idx,
            )
            start_idx += len(output_tokens)
        # ======

        # save scaffold info
        np.savez(
            os.path.join(scaffold_info_path, f"{ori_pdb_name}.npz"),
            start_idxs_list=start_idxs_list,
            end_idxs_list=end_idxs_list,
            scaffold_lengths_list=scaffold_lengths_list,
        )


def save_results(
    tokenizer,
    struct_tokenizer,
    save_dir,
    output_tokens,
    start_idx, # NOTE: add start_idx to count samples
    headers=None,
    save_pdb=False,
    continue_write=False,
):
    # save to fasta
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}...")
    if headers is None:
        headers = [f"sample_{i}" for i in range(start_idx, start_idx + len(output_tokens))] # NOTE: add start_idx to count samples

    struct_tokens, aatype_tokens = output_tokens.chunk(2, dim=-1)
    aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
    struct_tokens_strings = list(
        map(
            lambda s: ",".join(s.split()),
            tokenizer.batch_decode(struct_tokens, skip_special_tokens=True),
        )
    )
    aatype_strings = list(
        map(
            lambda s: "".join(s.split()),
            tokenizer.batch_decode(aatype_tokens, skip_special_tokens=True),
        )
    )
    save_fasta(
        save_name=aatype_fasta_path,
        output_results=aatype_strings,
        headers=headers,
        continue_write=continue_write,
    )
    if save_pdb:
        pdb_save_dir = os.path.join(save_dir, "decoded_pdb")
        os.makedirs(pdb_save_dir, exist_ok=True)
        for header, aatype_str, struct_tokens_str in zip(
            headers, aatype_strings, struct_tokens_strings
        ):
            (
                aatype_tensor,
                struct_tokens_tensor,
            ) = struct_tokenizer.string_to_tensor(
                aatype_str, struct_tokens_str
            )
            decoder_out = struct_tokenizer.detokenize(struct_tokens_tensor)
            decoder_out["aatype"] = aatype_tensor
            decoder_out["header"] = [header]

            struct_tokenizer.output_to_pdb(
                decoder_out, output_dir=pdb_save_dir
            )

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seqs", type=int, default=20)
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--saveto", type=str, default="gen.fasta")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy", type=str, default="annealing@2.0:1.0"
    )
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--model_name", type=str, default="airkingbd/dplm2_650m"
    )
    parser.add_argument(
        "--motif_aa",
        type=str,
        default="./data-bin/scaffolding-pdbs/aa_seq.fasta",
    )
    parser.add_argument(
        "--motif_struct",
        type=str,
        default="./data-bin/scaffolding-pdbs/struct_seq.fasta",
    )

    # ======
    # NOTE: add unmasking strategy, maxiter_seqlen_ratio, and bit_model for ablation study
    parser.add_argument(
        "--unmasking_strategy", type=str, default="stochastic1.0"
    )
    parser.add_argument("--maxiter_seqlen_ratio", type=int, default=1)
    parser.add_argument("--bit_model", action="store_true")
    # ======

    # ======
    # NOTE: CFG options — split cfg_scale per track
    parser.add_argument("--seq_cfg_scale", type=float, default=0.0,
                        help="CFG scale for sequence track. 0 = no guidance.")
    parser.add_argument("--struct_cfg_scale", type=float, default=0.0,
                        help="CFG scale for structure track. 0 = no guidance.")
    parser.add_argument("--cfg_mode", type=str, default="drop_motif",
                        choices=["drop_condition", "drop_motif", "full_mask", "cross_modal", "motif_anchored_cross_modal", "drop_motif_cross_modal"],
                        help="How to construct unconditional input for CFG.")
    # ======

    args = parser.parse_args()
    pprint(args)

    seed_everything(args.seed, workers=True)

    generate(args, args.saveto)


if __name__ == "__main__":
    main()
