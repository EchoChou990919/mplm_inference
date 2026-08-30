# NOTE: This file was created for this mplm_inference repository
set -euo pipefail

: "${PROJECT_ROOT:?Please export PROJECT_ROOT as the repository root first.}"
model_name=esm3-open

seed=42

seq_num_steps=-1
seq_protlen_steps=1
seq_temp=1.0
seq_temp_annealing=1
seq_unmasking_strategy=stochastic
seq_remasking=1

struct_num_steps=-1
struct_protlen_steps=1
struct_temp=0.7
struct_temp_annealing=0
struct_unmasking_strategy=random
struct_remasking=0

ss_num_steps=-1
ss_protlen_steps=1
ss_temp=0.7
ss_temp_annealing=0
ss_unmasking_strategy=random

seq_struct_order=cogen

if [ "$seq_struct_order" = "cogen" ]; then
    struct_num_steps=$seq_num_steps
    struct_protlen_steps=$seq_protlen_steps
    struct_temp=$seq_temp
    struct_temp_annealing=$seq_temp_annealing
    struct_unmasking_strategy=$seq_unmasking_strategy
    struct_remasking=$seq_remasking
fi

output_dir="generation-results/${model_name}/vanilla_uncon_cogen"
if [ -d "${output_dir}" ]; then
    echo "${output_dir} exists; skip generation."
else
    echo "running for ${output_dir}"
    mkdir -p "${output_dir}"

    python esm3_uncon_cogen.py --seed $seed \
        --model_name $model_name \
        --seq_num_steps $seq_num_steps \
        --seq_protlen_steps $seq_protlen_steps \
        --seq_temp $seq_temp \
        --seq_temp_annealing $seq_temp_annealing \
        --seq_unmasking_strategy $seq_unmasking_strategy \
        --seq_remasking $seq_remasking \
        --struct_num_steps $struct_num_steps \
        --struct_protlen_steps $struct_protlen_steps \
        --struct_temp $struct_temp \
        --struct_temp_annealing $struct_temp_annealing \
        --struct_unmasking_strategy $struct_unmasking_strategy \
        --struct_remasking $struct_remasking \
        --ss_num_steps $ss_num_steps \
        --ss_protlen_steps $ss_protlen_steps \
        --ss_temp $ss_temp \
        --ss_temp_annealing $ss_temp_annealing \
        --ss_unmasking_strategy $ss_unmasking_strategy \
        --output_root ${output_dir} \
        --seq_struct_order $seq_struct_order \
        --disable_tqdm
fi

# ESMFold structure prediction
conda run -n dplm2 python "${PROJECT_ROOT}/evals/cal_plddt_dir.py" -i "${output_dir}"

conda run -n dplm2 python "${PROJECT_ROOT}/evals/eval_uncon_cogen.py" \
    --results_root "${output_dir}" \
    --designability --abc_ratio --diversity
