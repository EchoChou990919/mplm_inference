# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=esm3-open

seed=42

struct_num_steps=8
struct_protlen_steps=-1
struct_temp=0.5
unmasking_strategy=stochastic
annealing=1
remasking=1

cfg_scale=2.0
cfg_mode="drop_condition"

reward_mode="ptm"

beam_width=4
branching_factor=4
scoring_interval=1
num_scoring_rounds=0

for dataset in 'cameo2022' 'pdb_date'; do
    input_fasta_path=${PROJECT_ROOT}/data/${dataset}/aatype.fasta
    output_dir="generation-results/${model_name}/rewarded_folding/${dataset}"
    if [ -d "${output_dir}" ]; then
        echo "${output_dir} exists; skip generation."
    else
        echo "running for ${output_dir}"
        mkdir -p "${output_dir}"

        python esm3_rewarded_folding.py --seed $seed \
            --struct_num_steps $struct_num_steps \
            --struct_protlen_steps $struct_protlen_steps \
            --struct_temp $struct_temp \
            --temp_annealing $annealing \
            --unmasking_strategy $unmasking_strategy \
            --remasking $remasking \
            --model_name $model_name \
            --input_fasta_path ${input_fasta_path} \
            --output_dir ${output_dir} \
            --cfg_scale $cfg_scale \
            --cfg_mode $cfg_mode \
            --beam_width $beam_width \
            --branching_factor $branching_factor \
            --scoring_interval $scoring_interval \
            --num_scoring_rounds $num_scoring_rounds \
            --reward_mode $reward_mode \
            --disable_tqdm
    fi

    conda run -n dplm2 python "${PROJECT_ROOT}/evals/eval_for_inv_folding.py" \
        --task folding \
        --data_dir "${PROJECT_ROOT}/data/${dataset}" \
        --results_root "${output_dir}"
done
