# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=dplm2_650m

max_iter=500
maxiter_seqlen_ratio=-1
sampling_strategy=annealing@2.0:0.1
unmasking_strategy=stochastic1.0
batch_size=50
seed=42

cfg_mode=cross_modal
seq_cfg_scale=2.0
struct_cfg_scale=1.5

output_dir="generation-results/${model_name}/guided_uncon_cogen"
if [ -d "${output_dir}" ]; then
    echo "${output_dir} exists; skip generation."
else
    echo "running for ${output_dir}"
    mkdir -p "${output_dir}"

    python guided_generate_dplm2.py --seed ${seed} \
        --model_name airkingbd/${model_name} \
        --task co_generation \
        --max_iter ${max_iter} \
        --maxiter_seqlen_ratio ${maxiter_seqlen_ratio} \
        --sampling_strategy ${sampling_strategy} \
        --unmasking_strategy ${unmasking_strategy} \
        --seq_lens 100 200 300 400 500 \
        --num_seqs 100 \
        --batch_size ${batch_size} \
        --saveto ${output_dir} \
        --seq_cfg_scale ${seq_cfg_scale} \
        --struct_cfg_scale ${struct_cfg_scale} \
        --cfg_mode ${cfg_mode}

    python scripts/reorganize_uncon_cogen_results.py --source_root ${output_dir}
fi

# ESMFold structure prediction
python "${PROJECT_ROOT}/evals/cal_plddt_dir.py" -i "${output_dir}"

python "${PROJECT_ROOT}/evals/eval_uncon_cogen.py" \
    --results_root "${output_dir}" \
    --designability --abc_ratio --diversity
