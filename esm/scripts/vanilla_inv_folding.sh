# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=esm3-open

seed=42

seq_num_steps=8
seq_protlen_steps=-1
seq_temp=0.5
unmasking_strategy=entropy
annealing=1
remasking=1

for dataset in 'cameo2022' 'pdb_date'; do
    input_dir=${PROJECT_ROOT}/data/${dataset}
    output_dir="generation-results/${model_name}/vanilla_inv_folding/${dataset}"
    if [ -d "${output_dir}" ]; then
        echo "${output_dir} exists; skip generation."
    else
        echo "running for ${output_dir}"
        mkdir -p "${output_dir}"

        python esm3_inv_folding.py --seed $seed \
            --seq_num_steps $seq_num_steps \
            --seq_protlen_steps $seq_protlen_steps \
            --seq_temp $seq_temp \
            --temp_annealing $annealing \
            --unmasking_strategy $unmasking_strategy \
            --remasking $remasking \
            --model_name $model_name \
            --input_dir ${input_dir} \
            --output_dir ${output_dir} \
            --disable_tqdm
    fi

    # ESMFold structure prediction
    conda run -n dplm2 python "${PROJECT_ROOT}/evals/cal_plddt_dir.py" -i "${output_dir}"

    conda run -n dplm2 python "${PROJECT_ROOT}/evals/eval_for_inv_folding.py" \
        --task inv_folding \
        --data_dir "${PROJECT_ROOT}/data/${dataset}" \
        --results_root "${output_dir}"
done
