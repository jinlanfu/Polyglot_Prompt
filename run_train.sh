export model_dir='./models'
export taskname="tydiqa,pawsx,xnli,mldoc,marc,mlqa,xquad"
export model_name="mt5base_polyprompt_crossLanguage"
export output_dir=${model_dir}/${model_name}
export model_path='google/mt5-base'

export datadir='./datas/'
export prompt_dir='./datas/templates/CL/'
export train_filename='train3k_promptCL_7datas_18w.pt'

export train_file_path=${datadir}/${train_filename}
export num_train_epochs=18
export save_steps=4000
export do_train=True # set do_train=False, if you don't need to fine-tuning the model.
export do_eval=True
export do_test=True
export PER_DEVICE_TRAIN_BATCH_SIZE=18
export PER_DEVICE_EVAL_BATCH_SIZE=2
export gradient_accumulation_steps=5
export eval_batch_size=100

CUDA_VISIBLE_DEVICES=2,3 nohup python ./train_mt5.py \
    --output_dir=$output_dir \
    --taskname=${taskname} \
    --model_name_or_path=$model_path \
    --train_file_path=$train_file_path \
    --overwrite_output_dir=True \
     --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
     --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
     --source_max_len=512 \
     --target_max_len=64 \
     --eval_batch_size=$eval_batch_size \
     --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --learning_rate=1e-4 \
    --num_train_epochs=$num_train_epochs \
    --save_steps=$save_steps \
    --do_train=$do_train \
    --do_eval=$do_eval \
    --data_dir=$datadir \
    --prompt_dir=$prompt_dir \
    --model_dir=$model_dir \
    >logs/TRAIN_model_[${model_name}]_taskname_[${taskname}]_2.txt &




