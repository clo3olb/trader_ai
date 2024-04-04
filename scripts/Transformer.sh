if [ ! -d "./predictor/logs" ]; then
    mkdir ./predictor/logs
fi

seq_len=336
pred_len=96
model_name=Transformer

root_path_name=./predictor/dataset/
data_path_name=$2.csv
model_id_name=$1
data_name=custom

echo $model_id_name $data_path_name

random_seed=2021
python -u predictor/run_experiment.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --date_header 'Timestamp'\
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $3 \
    --dec_in $3 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --c_out $3 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --stride 8\
    --decomposition 1 \
    --des 'Exp' \
    --train_epochs 100\
    --patience 20\
    --itr 1\
    --target 'Close'\
    --freq 'd' \
    --batch_size 256 --learning_rate 0.0001 >predictor/logs/$model_name'_'$model_id_name.log 