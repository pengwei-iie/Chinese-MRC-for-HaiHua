PER_GPU_BATCH_SIZE=2
BATCH_SIZE=32
MAX_LEN=128
GC_STEP=16
GPU_NUM=1
lr_iter=(1 2) #  (1 0.5 2)
epoch_iter=(5 6 7 8 9 10)
SEED=42 # seed_iter=(42 100 512 1024 2019)

task_iter=('task4sta') # TASK_NAME='task4stb'
NET_NAME='basic'
M_NAME='roberta_large'

BASE_DIR=/home/xingluxi/IIENLP-SemEval2021/src
DATA_DIR=/data1/data-xlx/semeval21_task4/SemEval2021-Task4
MODEL_PATH=/home/data1/data-xlx/pytorch-pretrain-lm/pytorch-roberta-large
TK_PATH=/home/data1/data-xlx/pytorch-pretrain-lm/pytorch-roberta-large

OUTPUT_BASE=/data1/data-xlx/semeval21_task4/checkpoints_sta

for TASK_NAME in ${task_iter[@]}; do
    for LR_RATE in ${lr_iter[@]}; do
        for EPOCH_N in ${epoch_iter[@]}; do

            EXP_PRE="seed${SEED}"

            OUTPUT_DIR=${OUTPUT_BASE}/${TASK_NAME}_${NET_NAME}_${M_NAME}_gn${GPU_NUM}_bp${PER_GPU_BATCH_SIZE}_gc${GC_STEP}_lr${LR_RATE}_l${MAX_LEN}_e${EPOCH_N}_${EXP_PRE}
            mkdir -p ${OUTPUT_DIR}

            CUDA_VISIBLE_DEVICES=1 \
            python -u run_bert_task4sta.py \
            --task_name ${TASK_NAME} \
            --data_dir ${DATA_DIR} \
            --num_choices 5 \
            --model_type roberta \
            --network_name ${NET_NAME} \
            --model_name_or_path ${MODEL_PATH} \
            --tokenizer_name_or_path ${TK_PATH} \
            --output_dir ${OUTPUT_DIR} \
            --log_file ${BASE_DIR}/log.${TASK_NAME}.${NET_NAME}.${M_NAME}.gn${GPU_NUM}.bp${PER_GPU_BATCH_SIZE}.gc${GC_STEP}.lr${LR_RATE}.l${MAX_LEN}.e${EPOCH_N}.${EXP_PRE}.out \
            --result_eval_file result.eval.${TASK_NAME}.${NET_NAME}.${M_NAME}.gn${GPU_NUM}.bp${PER_GPU_BATCH_SIZE}.gc${GC_STEP}.lr${LR_RATE}.l${MAX_LEN}.e${EPOCH_N}.${EXP_PRE}.txt \
            --tfboard_log_dir ${OUTPUT_DIR}/tfboard.event.out \
            --do_train \
            --do_eval \
            --do_trial \
            --have_test_label \
            --do_lower_case \
            --overwrite_output_dir \
            --max_seq_length ${MAX_LEN} \
            --train_batch_size ${BATCH_SIZE} \
            --per_gpu_train_batch_size ${PER_GPU_BATCH_SIZE} \
            --per_gpu_eval_batch_size 4 \
            --num_train_epochs ${EPOCH_N} \
            --max_steps -1 \
            --learning_rate ${LR_RATE}e-5 \
            --gradient_accumulation_steps ${GC_STEP} \
            --max_grad_norm 0.0 \
            --seed ${SEED}

        done
    done
done