# model_name, data_name, input_size
sh scripts/PatchTST.sh PatchTST_AAPL AAPL 5
sh scripts/PatchTST.sh PatchTST_AAPL_pct AAPL_pct 5

sh scripts/Autoformer.sh Autoformer_AAPL AAPL 5
sh scripts/Autoformer.sh Autoformer_AAPL_pct AAPL_pct 5

sh scripts/DLinear.sh DLinear_AAPL AAPL 5
sh scripts/DLinear.sh DLinear_AAPL_pct AAPL_pct 5
