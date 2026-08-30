# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=esm3-open

seed=42
prot_num=100

seq_num_steps=-1
seq_protlen_steps=1
seq_unmasking_strategy=random
seq_temp=1.0
seq_temp_annealing=1
seq_remasking=1

struct_num_steps=$seq_num_steps
struct_protlen_steps=$seq_protlen_steps
struct_temp=$seq_temp
struct_unmasking_strategy=$seq_unmasking_strategy
struct_temp_annealing=$seq_temp_annealing
struct_remasking=$seq_remasking

seq_cfg_scale=2.0
struct_cfg_scale=2.0
cfg_mode="drop_motif_cross_modal"

reward_alpha=1.0
reward_beta=1.0
seq_reward_threshold=0.5
struct_reward_threshold=0.5
selection_mode="best"
seq_reward_type="foldability"
reward_mode="ptm"

beam_width=4
branching_factor=1
scoring_interval=0
num_scoring_rounds=5

data_dir=${PROJECT_ROOT}/data/scaffolding_pdbs/ori_pdbs
output_dir="generation-results/${model_name}/rewarded_motif_scaffolding"
if [ -d "${output_dir}" ]; then
    echo "${output_dir} exists; skip generation."
else
    echo "running for ${output_dir}"
    mkdir -p "${output_dir}"

    python esm3_rewarded_motif_scaffolding.py --seed $seed \
        --seq_num_steps $seq_num_steps \
        --seq_protlen_steps $seq_protlen_steps \
        --seq_temp $seq_temp \
        --seq_unmasking_strategy $seq_unmasking_strategy \
        --seq_temp_annealing $seq_temp_annealing \
        --seq_remasking $seq_remasking \
        --struct_num_steps $struct_num_steps \
        --struct_protlen_steps $struct_protlen_steps \
        --struct_temp $struct_temp \
        --struct_unmasking_strategy $struct_unmasking_strategy \
        --struct_temp_annealing $struct_temp_annealing \
        --struct_remasking $struct_remasking \
        --model_name $model_name \
        --data_dir ${data_dir} \
        --saveto ${output_dir} \
        --prot_num $prot_num \
        --seq_cfg_scale $seq_cfg_scale \
        --struct_cfg_scale $struct_cfg_scale \
        --cfg_mode $cfg_mode \
        --beam_width $beam_width \
        --branching_factor $branching_factor \
        --scoring_interval $scoring_interval \
        --num_scoring_rounds $num_scoring_rounds \
        --reward_alpha $reward_alpha \
        --reward_beta $reward_beta \
        --seq_reward_type $seq_reward_type \
        --reward_mode $reward_mode \
        --seq_reward_threshold $seq_reward_threshold \
        --struct_reward_threshold $struct_reward_threshold \
        --selection_mode $selection_mode \
        --disable_tqdm
fi

conda run -n dplm2 python "${PROJECT_ROOT}/evals/cal_plddt_dir.py" -i "${output_dir}"

conda run -n dplm2 python "${PROJECT_ROOT}/evals/eval_scaffolding.py" \
    --data_dir "${PROJECT_ROOT}/data/scaffolding_pdbs" \
    --results_root "${output_dir}" \
    --prefix struct
