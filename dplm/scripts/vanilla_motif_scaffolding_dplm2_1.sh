# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=dplm2_bit_650m

max_iter=500
maxiter_seqlen_ratio=-1
unmasking_strategy=stochastic1.0
sampling_strategy=annealing@0.5:0.0
batch_size=50
seed=42

output_dir="generation-results/${model_name}/vanilla_motif_scaffolding"
if [ -d "${output_dir}" ]; then
    echo "${output_dir} exists; skip generation."
else
    echo "running for ${output_dir}"
    mkdir -p "${output_dir}"

    python scaffold_generate_dplm2.py --seed ${seed} \
        --model_name airkingbd/${model_name} \
        --motif_aa ${PROJECT_ROOT}/data/scaffolding_pdbs/aa_seq.fasta \
        --motif_struct ${PROJECT_ROOT}/data/scaffolding_pdbs/struct_seq.fasta \
        --num_seqs 100 \
        --batch_size ${batch_size} \
        --unmasking_strategy ${unmasking_strategy} \
        --sampling_strategy ${sampling_strategy} \
        --max_iter ${max_iter} \
        --maxiter_seqlen_ratio ${maxiter_seqlen_ratio} \
        --saveto ${output_dir} \
        --bit_model

    python scripts/reorganize_scaffolding_results.py --source_root ${output_dir}
fi

python "${PROJECT_ROOT}/evals/cal_plddt_dir.py" -i "${output_dir}"

python "${PROJECT_ROOT}/evals/eval_scaffolding.py" \
    --data_dir "${PROJECT_ROOT}/data/scaffolding_pdbs" \
    --results_root "${output_dir}"
