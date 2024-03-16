if [ ! -d "./predictor/logs" ]; then
    mkdir ./predictor/logs
fi

seq_len=336
model_name=PatchTST

root_path_name=./predictor/dataset/
data_path_name=AAPL_intraday.csv
model_id_name=AAPL_intraday
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
      --date_header 'Date'\
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 19 \
      --dec_in 19 \
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
      --target 'Close'\
      --batch_size 512 --learning_rate 0.0001 >predictor/logs/$model_name'_'$model_id_name.log 
done