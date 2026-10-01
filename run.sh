#!/bin/bash
# ================= 配置区域 =================
# 运行哪一个脚本：domains（生成域标签）/ optuna（嵌套 CV 搜参）
#                 evaluate（固定超参六折评估）/ plot（t-SNE 与 Wasserstein）
STAGE=${1:-optuna}

# 要处理的骨干模型（可只填一部分）
MODELS="AttentionBaseNet EEGMiner EEGNet BrainModule EEGConformer BIOT FBCNet IFNet MSVTNet SSTDPN"

# 搜参的 trial 数与训练轮数
TRIALS=50
EPOCHS=100

# 使用的显卡
GPUS=(1 2 3)
# ===========================================

pids=()
cleanup() {
    echo -e "\nCaught exit signal. Killing all child processes..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null; fi
    done
    exit 1
}
trap cleanup INT TERM

if [ "$STAGE" = "domains" ]; then
    echo "Building the fixed domain labels ..."
    python domains.py
    exit 0
fi

# 逐个骨干、逐个搜索组排队；同一骨干的三个搜索组必须按 lr -> clf/var 的顺序
queue=()
for model in $MODELS; do
    case "$STAGE" in
        optuna)   queue+=("$model:lr" "$model:clf" "$model:var") ;;
        evaluate) for c in full no_subj_grl_var no_domain_grl_bce baseline; do
                      queue+=("$model:$c"); done ;;
        plot)     queue+=("$model:plot") ;;
    esac
done

echo "Total jobs: ${#queue[@]} ; GPUs: ${GPUS[*]}"

i=0
for job in "${queue[@]}"; do
    model="${job%%:*}"
    task="${job##*:}"
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    case "$STAGE" in
        optuna)   cmd="python optuna.py --model $model --group $task --trials $TRIALS --epochs $EPOCHS --gpu $gpu" ;;
        evaluate) cmd="python evaluate.py --model $model --condition $task --epochs $EPOCHS --gpu $gpu" ;;
        plot)     cmd="python plot.py --model $model --epochs $EPOCHS --gpu $gpu" ;;
    esac
    log_file="log_${STAGE}_${model}_${task}_gpu${gpu}.txt"
    echo "Launching: $cmd"
    CUDA_VISIBLE_DEVICES=$gpu $cmd > "$log_file" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    # 每张卡同时只跑一个任务
    if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done

wait
echo "--- All $STAGE tasks finished. ---"
trap - INT TERM
