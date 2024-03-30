if [ ! -d "./predictor/logs" ]; then
    mkdir ./predictor/logs
fi

seq_len=336
model_name=DLinear

root_path_name=./predictor/dataset/
data_path_name=AAPL.csv
model_id_name=AAPL
data_name=custom

random_seed=2021
# for pred_len in 96 192 336 720
for pred_len in 96
do
  python -u predictor/run_experiment.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --date_header 'timestamp'\
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in 5 \
    --des 'Exp' \
    --itr 1 \
    --target 'Close'\
    --batch_size 256 \
    > predictor/logs/$model_name'_'$model_id_name.log 
done