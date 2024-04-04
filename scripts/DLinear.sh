if [ ! -d "./predictor/logs" ]; then
    mkdir ./predictor/logs
fi

seq_len=336
pred_len=96
model_name=DLinear

root_path_name=./predictor/dataset/
data_path_name=$2.csv
model_id_name=$1
data_name=custom

random_seed=2021
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
  --pred_len $pred_len \
  --enc_in $3 \
  --des 'Exp' \
  --itr 1 \
  --target 'Close'\
  --batch_size 256 \
  --patience 100 > predictor/logs/$model_name'_'$model_id_name.log 