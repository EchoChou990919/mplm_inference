# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=dplm2_bit_650m

maxiter_seqlen_ratio=-1
batch_size=50
unmasking_strategy=deterministic
max_iter=8
sampling_strategy=annealing@0.1:0.0
seed=42

for dataset in 'cameo2022' 'pdb_date'; do
    input_fasta_path=${PROJECT_ROOT}/data/${dataset}/aatype.fasta
    output_dir="generation-results/${model_name}/vanilla_folding/${dataset}"
    if [ -d "${output_dir}" ]; then
        echo "${output_dir} exists; skip generation."
    else
        echo "running for ${output_dir}"
        mkdir -p "${output_dir}"

        python generate_dplm2.py --seed ${seed} \
            --model_name airkingbd/${model_name} \
            --task folding \
            --input_fasta_path ${input_fasta_path} \
            --max_iter ${max_iter} \
            --maxiter_seqlen_ratio ${maxiter_seqlen_ratio} \
            --batch_size ${batch_size} \
            --unmasking_strategy ${unmasking_strategy} \
            --sampling_strategy ${sampling_strategy} \
            --saveto ${output_dir} \
            --bit_model
    fi

    rm -f "${output_dir}/aatype.fasta"

    python "${PROJECT_ROOT}/evals/eval_for_inv_folding.py" \
        --task folding \
        --data_dir "${PROJECT_ROOT}/data/${dataset}" \
        --results_root "${output_dir}"
done
