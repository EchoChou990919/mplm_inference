# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=esm3-open

seed=42

seq_num_steps=-1
seq_protlen_steps=1
seq_temp=0.7
seq_unmasking_strategy=random
seq_temp_annealing=1
seq_remasking=1

struct_num_steps=-1
struct_protlen_steps=8
struct_temp=1.0
struct_unmasking_strategy=random
struct_temp_annealing=1
struct_remasking=0

seq_struct_order=cogen

seq_cfg_scale=2.0
struct_cfg_scale=2.0
cfg_mode="drop_motif_cross_modal"

if [ "$seq_struct_order" = "cogen" ]; then
    struct_num_steps=$seq_num_steps
    struct_protlen_steps=$seq_protlen_steps
    struct_temp=$seq_temp
    struct_unmasking_strategy=$seq_unmasking_strategy
    struct_temp_annealing=$seq_temp_annealing
    struct_remasking=$seq_remasking
fi

data_dir=${PROJECT_ROOT}/data/scaffolding_pdbs/ori_pdbs
output_dir="generation-results/${model_name}/guided_motif_scaffolding"
if [ -d "${output_dir}" ]; then
    echo "${output_dir} exists; skip generation."
else
    echo "running for ${output_dir}"
    mkdir -p "${output_dir}"

    python esm3_cfg_motif_scaffolding.py --seed $seed \
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
        --seq_struct_order $seq_struct_order \
        --seq_cfg_scale $seq_cfg_scale \
        --struct_cfg_scale $struct_cfg_scale \
        --cfg_mode $cfg_mode \
        --disable_tqdm
fi

conda run -n dplm2 python "${PROJECT_ROOT}/evals/cal_plddt_dir.py" -i "${output_dir}"

conda run -n dplm2 python "${PROJECT_ROOT}/evals/eval_scaffolding.py" \
    --data_dir "${PROJECT_ROOT}/data/scaffolding_pdbs" \
    --results_root "${output_dir}" \
    --prefix struct
