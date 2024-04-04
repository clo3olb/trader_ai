# model_name, data_name, input_size


# # Sin Wave
# sh scripts/PatchTST.sh PatchTST_sin_wave sin_wave.csv 5
# sh scripts/DLinear.sh DLinear_sin_wave sin_wave.csv 5
# sh scripts/Transformer.sh Transformer_sin_wave sin_wave.csv 5
# sh scripts/Autoformer.sh Autoformer_sin_wave sin_wave.csv 5

# # AAPL
# sh scripts/PatchTST.sh PatchTST_AAPL AAPL.csv 5
# sh scripts/DLinear.sh DLinear_AAPL AAPL.csv 5
# sh scripts/Autoformer.sh Autoformer_AAPL AAPL.csv 5
# sh scripts/Transformer.sh Transformer_AAPL AAPL.csv 5


# # AAPL_pct
# sh scripts/PatchTST.sh PatchTST_AAPL_pct AAPL_pct 5
# sh scripts/DLinear.sh DLinear_AAPL_pct AAPL_pct 5
# sh scripts/Autoformer.sh Autoformer_AAPL_pct AAPL_pct 5
# sh scripts/Transformer.sh Transformer_AAPL_pct AAPL_pct 5

symbols=("MSFT" "AMZN" "GOOGL" "NFLX"
        "JPM" "BAC" "C" "WFC" "GS"\
        "KO" "PG" "MCD" "DIS" "NKE"\
        "JNJ" "PFE" "MRK" "ABT" "BMY"\
        "XOM" "CVX" "COP" "SLB" "PSX")
# other symbols
for symbol in "${symbols[@]}"
do
    sh scripts/PatchTST.sh PatchTST_$symbol $symbol.csv 5
    sh scripts/PatchTST.sh PatchTST_${symbol}_pct ${symbol}_pct.csv 5

done