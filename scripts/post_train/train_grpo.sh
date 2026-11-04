# 公共环境变量
if [ -f scripts/utils/env.sh ]; then
    source scripts/utils/env.sh
fi
export PYTHONPATH=`pwd`
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_DEBUG=DEBUG
#export NCCL_BLOCKING_WAIT=0
export NCCL_IB_TIMEOUT=22
#export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_TIMEOUT=600
export GLOO_SOCKET_IFNAME=bond1
export TOKENIZERS_PARALLELISM=false

echo "--------------------------------------------------------------------------"
#################################PARAMS setup#####################################
# Optional VLM roots / presets (see hyvideo/.../vlm_presets.py). Unset to use code defaults.
# export PUBLIC_VLM_ROOT="/path/to/1_public_models"
# export EMO_REWARD_VLM_PRESET="qwen2-vl-2b"
# export VIDEOALIGN_VLM_PRESET="qwen2-vl-2b"
# export EMO_REWARD_COMBINE_VIDEOALIGN="true"
# export VIDEOALIGN_FUSION_ALPHA="0.5"
# export VIDEOALIGN_FUSION_BETA="0.5"

export EMO_REWARD_VLM_PRESET="${EMO_REWARD_VLM_PRESET:-qwen3-vl-30b-a3b}"
export VIDEOALIGN_VLM_PRESET="${VIDEOALIGN_VLM_PRESET:-qwen2-vl-2b}"
# Leave unset to resolve presets to public Hugging Face ids; point it at a local
# directory of pre-downloaded weights to use those instead (see vlm_presets.py).
export PUBLIC_VLM_ROOT="${PUBLIC_VLM_ROOT:-}"
export EMO_REWARD_PLAN_ENABLED="${EMO_REWARD_PLAN_ENABLED:-False}"
export EMO_REWARD_PLAN_BACKEND="${EMO_REWARD_PLAN_BACKEND:-gemini}"
export EMO_REWARD_PLAN_GEMINI_MODEL="${EMO_REWARD_PLAN_GEMINI_MODEL:-gemini-2.5-flash}"

export EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SP_SIZE="${SP_SIZE:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export VALIDATE_AT_STEP0="${VALIDATE_AT_STEP0:-False}"
export ENABLE_EMOTION_SCORERS="${ENABLE_EMOTION_SCORERS:-True}"
export CLIP_AUTT_WEIGHT="${CLIP_AUTT_WEIGHT:-0.0}"
export R1OMNI_WEIGHT="${R1OMNI_WEIGHT:-0.15}"
export R1OMNI_VAD_MODE="${R1OMNI_VAD_MODE:-alignment}"
export ENABLE_CHROMATIC="${ENABLE_CHROMATIC:-True}"
export ENABLE_KINETIC="${ENABLE_KINETIC:-True}"
export ENABLE_MOTION_NATURALNESS="${ENABLE_MOTION_NATURALNESS:-True}"
export MOTION_NATURALNESS_WEIGHT="${MOTION_NATURALNESS_WEIGHT:-0.20}"
export MOTION_NATURALNESS_TARGET_AROUSAL="${MOTION_NATURALNESS_TARGET_AROUSAL:-True}"
export ADAPTIVE_REFLECT_CONTROL="${ADAPTIVE_REFLECT_CONTROL:-True}"
export REFLECT_WARMUP_START_STEP="${REFLECT_WARMUP_START_STEP:-40}"
export REFLECT_FULL_ON_STEP="${REFLECT_FULL_ON_STEP:-100}"
export REFLECT_CONFLICT_SUPPRESSION="${REFLECT_CONFLICT_SUPPRESSION:-True}"
export REFLECT_SUBTLE_PROMPT_GATING="${REFLECT_SUBTLE_PROMPT_GATING:-True}"
export REWARD_GATE_POLICY="${REWARD_GATE_POLICY:-adaptive}"
export REWARD_GATE_SINGLE_TOOL="${REWARD_GATE_SINGLE_TOOL:-r1omni}"
export REWARD_GATE_JIT_TABLE="${REWARD_GATE_JIT_TABLE:-}"
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

GATE_EXTRA_ARGS=""
if [[ -n "${REWARD_GATE_JIT_TABLE:-}" ]]; then
  GATE_EXTRA_ARGS="${GATE_EXTRA_ARGS} --reward_gate_jit_table ${REWARD_GATE_JIT_TABLE}"
fi

VLM_EXTRA_ARGS=""
if [[ -n "${PUBLIC_VLM_ROOT:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --public_vlm_root ${PUBLIC_VLM_ROOT}"
fi
if [[ -n "${EMO_REWARD_VLM_PRESET:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --emo_reward_vlm_preset ${EMO_REWARD_VLM_PRESET}"
fi
if [[ -n "${VIDEOALIGN_VLM_PRESET:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --videoalign_vlm_preset ${VIDEOALIGN_VLM_PRESET}"
fi
if [[ -n "${EMO_REWARD_PLAN_ENABLED:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --emo_reward_plan_enabled ${EMO_REWARD_PLAN_ENABLED}"
fi
if [[ -n "${EMO_REWARD_PLAN_BACKEND:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --emo_reward_plan_backend ${EMO_REWARD_PLAN_BACKEND}"
fi
if [[ -n "${EMO_REWARD_PLAN_GEMINI_MODEL:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --emo_reward_plan_gemini_model ${EMO_REWARD_PLAN_GEMINI_MODEL}"
fi
if [[ -n "${EMO_REWARD_COMBINE_VIDEOALIGN:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --emo_reward_combine_videoalign ${EMO_REWARD_COMBINE_VIDEOALIGN}"
fi
if [[ -n "${VIDEOALIGN_FUSION_ALPHA:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --videoalign_fusion_alpha ${VIDEOALIGN_FUSION_ALPHA}"
fi
if [[ -n "${VIDEOALIGN_FUSION_BETA:-}" ]]; then
  VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS} --videoalign_fusion_beta ${VIDEOALIGN_FUSION_BETA}"
fi

if [[ "${CURRENT_TIME}" = "" ]]; then
    CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")
fi

if [[ "${START_EXPR_TIME}" = "" ]]; then
    START_EXPR_TIME=${CURRENT_TIME}
fi

output_dir=SAGE_GRPO_results/$START_EXPR_TIME
if [[ -n "${EXPERIMENT_TAG}" ]]; then
    output_dir="${output_dir}_${EXPERIMENT_TAG}"
fi

data_num_workers=8

ckpt_cnt=`ls -d ${output_dir}/checkpoints/global_step-* 2>/dev/null | sort -t- -k2,2 -n -r | wc -l`
if [[ ${ckpt_cnt} -ne 0 ]]; then
    ckpt_dir=`cd ${output_dir}/checkpoints/; ls -d global_step-* | sort -t- -k2,2 -n -r | head -1`
    resume="${output_dir}/checkpoints/${ckpt_dir}"
    echo "there exists ckpt dir, set the resume=${ckpt_dir}"
else
    echo "there exists no ckpt dir, use config resume=${resume}"
fi

if [ -n "$resume" ] && [ "$resume" != "" ]; then
    training_params="$training_params --resume $resume"
fi

mkdir -p $output_dir/logs
node_rank=$INDEX
RANK_ID=${INDEX:-0}
CURRENT_LOG_FILE=$output_dir/logs/training_rank_${RANK_ID}.log
if [[ "${AUTO_RESUME_TRAIN}" = "true" ]]; then
    line_no=`grep -nw "${LOCAL_IP}" ${HOST_PATH} | cut -d: -f1`
    if [[ "${line_no}" != ""  ]]; then
        node_rank=$((${line_no} - 1))
        echo "get the local ip ${LOCAL_IP} node_rank ${node_rank} for resume train"
    else
        echo "fail to get the ${LOCAL_IP} node_rank for resume train!!!"
        exit 1
    fi
    CURRENT_LOG_FILE="$output_dir/logs/${CURRENT_TIME}_rank${node_rank}.log"
fi

TCCL_DATA_DIR="${TCCL_DATA_DIR:-/dockerdata/.tccl}"
TCCL_DATA_FILE="${TCCL_DATA_DIR}/tccl.data"
rm -f "${TCCL_DATA_FILE}"
if [[ $node_rank = 0 ]]; then
    if [ -d "${TCCL_DATA_DIR}" ]; then
        echo "tccl directory ${TCCL_DATA_DIR} exists, use it!"
    else
        mkdir -p ${TCCL_DATA_DIR}
        echo "tccl directory ${TCCL_DATA_DIR} does not exist, create it!"
    fi

    if [[ "${AUTO_RESUME_TRAIN}" != "true" ]]; then
        echo "${NODE_IP_LIST}" | awk -F: -v RS=, '{gsub(/[0-9]+$/, ""); ips = ips ? ips "," : ""; ips = ips "\047" $1 "\047"} END{print "workers[" ips "]"}' >> ${TCCL_DATA_FILE}
    else
        awk 'BEGIN{printf "workers["} {printf "%s\047%s\047", (NR>1?",":""),$1} END{print "]"} ' ${HOST_PATH} >> ${TCCL_DATA_FILE}
    fi
    echo "--master_addr=${CHIEF_IP}" >> ${TCCL_DATA_FILE}
    echo "--master_port=37294" >> ${TCCL_DATA_FILE}
fi

if [ -n "$config_file" ] && [ -n "$config_index" ]; then
python3 -c "
import yaml
import sys

config_file = sys.argv[1]
config_index = int(sys.argv[2])
output_file = sys.argv[3]

with open(config_file, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

if 'configs' in data and config_index < len(data['configs']):
    extracted_config = {'configs': [data['configs'][config_index]]}
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(extracted_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
else:
    print(f'Error: Config index {config_index} not found', file=sys.stderr)
    sys.exit(1)
" "$config_file" "$config_index" "$output_dir/$(basename $config_file)"
fi

torchrun --nnodes $HOST_NUM --nproc_per_node $HOST_GPU_NUM \
    --node_rank $node_rank \
    --rdzv_endpoint $CHIEF_IP:31959 \
    --rdzv_id 456 \
    post_train.py \
    --pretrained_model_root ./ckpts \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --num_generations ${NUM_GENERATIONS} \
    --max_steps ${MAX_STEPS} \
    --output_dir ${output_dir} \
    --enable_fsdp \
    --enable_gradient_checkpointing \
    --sp_size ${SP_SIZE} \
    --reward_model "${REWARD_MODEL:-emo_reward}" \
    --reward_checkpoint_mode "v3" \
    --save_interval ${SAVE_INTERVAL} \
    --validate_at_step0 ${VALIDATE_AT_STEP0} \
    --validate_video_length 121 \
    --validation_timestep_shift 5.0 \
    --use_grad_balancing True \
    --enable_timestep_permutation True \
    --sde_type "${SDE_TYPE:-sage_grpo}" \
    --kl_weight 1e-5 \
    --kl_coef 1e-7 \
    --use_moving_KL True \
    --update_ref_model_step 10 \
    --use_dual_kl True \
    --dual_kl_moving_weight 1.0 \
    --dual_kl_step_weight 0.1 \
    --reference_mode_offload True \
    --enable_emotion_scorers ${ENABLE_EMOTION_SCORERS} \
    --clip_autt_weight ${CLIP_AUTT_WEIGHT} \
    --r1omni_weight ${R1OMNI_WEIGHT} \
    --r1omni_vad_mode ${R1OMNI_VAD_MODE} \
    --enable_chromatic ${ENABLE_CHROMATIC} \
    --enable_kinetic ${ENABLE_KINETIC} \
    --enable_motion_naturalness ${ENABLE_MOTION_NATURALNESS} \
    --motion_naturalness_weight ${MOTION_NATURALNESS_WEIGHT} \
    --motion_naturalness_target_arousal ${MOTION_NATURALNESS_TARGET_AROUSAL} \
    --adaptive_reflect_control ${ADAPTIVE_REFLECT_CONTROL} \
    --reflect_warmup_start_step ${REFLECT_WARMUP_START_STEP} \
    --reflect_full_on_step ${REFLECT_FULL_ON_STEP} \
    --reflect_conflict_suppression ${REFLECT_CONFLICT_SUPPRESSION} \
    --reflect_subtle_prompt_gating ${REFLECT_SUBTLE_PROMPT_GATING} \
    --reward_gate_policy ${REWARD_GATE_POLICY} \
    --reward_gate_single_tool ${REWARD_GATE_SINGLE_TOOL} \
    ${GATE_EXTRA_ARGS} \
    ${training_params} \
    ${VLM_EXTRA_ARGS} \
    ${TRAIN_EXTRA_ARGS} \
    2>&1 | tee ${CURRENT_LOG_FILE}
