# model_name, data_name, input_size

# Sin Wave
# sh scripts/PatchTST.sh PatchTST_sin_wave sin_wave 5
sh scripts/Autoformer.sh Autoformer_sin_wave sin_wave 5
# sh scripts/DLinear.sh DLinear_sin_wave sin_wave 5

# AAPL
# sh scripts/PatchTST.sh PatchTST_AAPL AAPL 5
# sh scripts/Autoformer.sh Autoformer_AAPL AAPL 5
# sh scripts/DLinear.sh DLinear_AAPL AAPL 5

# AAPL_pct
# sh scripts/PatchTST.sh PatchTST_AAPL_pct AAPL_pct 5
# sh scripts/Autoformer.sh Autoformer_AAPL_pct AAPL_pct 5
# sh scripts/DLinear.sh DLinear_AAPL_pct AAPL_pct 5

# AAPL_intraday
# sh scripts/PatchTST.sh PatchTST_AAPL_intraday AAPL_intraday 19
# sh scripts/Autoformer.sh Autoformer_AAPL_intraday AAPL_intraday 19
# sh scripts/DLinear.sh DLinear_AAPL_intraday AAPL_intraday 19