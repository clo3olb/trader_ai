if [ ! -d "./src/logs" ]; then
    mkdir ./src/logs
fi

if [ ! -d "./src/logs/LongForecasting" ]; then
    mkdir ./src/logs/LongForecasting
fi
seq_len=336
model_name=PatchTST

root_path_name=./src/dataset/
data_path_name=AAPL.csv
model_id_name=AAPL
data_name=custom

random_seed=2021
# for pred_len in 96 192 336 720
for pred_len in 96
do
    python -u src/predict.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --itr 1\
      --checkpoints ./src/checkpoints/\
      --batch_size 256 --learning_rate 0.0001 >src/logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done