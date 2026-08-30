# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=esm3-open

seed=42

seq_num_steps=8
seq_protlen_steps=-1
unmasking_strategy="stochastic"
seq_temp=0.5
annealing=1
remasking=1

cfg_scale=2.0
cfg_mode="drop_condition"

reward_type="foldability"
reward_mode="ptm"

beam_width=4
branching_factor=4
scoring_interval=1
num_scoring_rounds=0

for dataset in 'cameo2022' 'pdb_date'; do
    input_dir=${PROJECT_ROOT}/data/${dataset}
    output_dir="generation-results/${model_name}/rewarded_inv_folding/${dataset}"
    if [ -d "${output_dir}" ]; then
        echo "${output_dir} exists; skip generation."
    else
        echo "running for ${output_dir}"
        mkdir -p "${output_dir}"

        python esm3_rewarded_inv_folding.py --seed $seed \
            --seq_num_steps $seq_num_steps \
            --seq_protlen_steps $seq_protlen_steps \
            --seq_temp $seq_temp \
            --temp_annealing $annealing \
            --unmasking_strategy $unmasking_strategy \
            --remasking $remasking \
            --model_name $model_name \
            --input_dir ${input_dir} \
            --output_dir ${output_dir} \
            --cfg_scale $cfg_scale \
            --cfg_mode $cfg_mode \
            --beam_width $beam_width \
            --branching_factor $branching_factor \
            --scoring_interval $scoring_interval \
            --num_scoring_rounds $num_scoring_rounds \
            --reward_type $reward_type \
            --reward_mode $reward_mode \
            --disable_tqdm
    fi

    # ESMFold structure prediction
    conda run -n dplm2 python "${PROJECT_ROOT}/evals/cal_plddt_dir.py" -i "${output_dir}"

    conda run -n dplm2 python "${PROJECT_ROOT}/evals/eval_for_inv_folding.py" \
        --task inv_folding \
        --data_dir "${PROJECT_ROOT}/data/${dataset}" \
        --results_root "${output_dir}"
done
