# model_name, data_name, input_size

set -e


# # Sin Wave
# sh scripts/PatchTST.sh PatchTST_sin_wave sin_wave.csv 19
# sh scripts/DLinear.sh DLinear_sin_wave sin_wave.csv 19
# sh scripts/Transformer.sh Transformer_sin_wave sin_wave.csv 19
# sh scripts/Autoformer.sh Autoformer_sin_wave sin_wave.csv 19

# # AAPL
# sh scripts/PatchTST.sh PatchTST_AAPL AAPL.csv 19
# sh scripts/DLinear.sh DLinear_AAPL AAPL.csv 19
# sh scripts/Autoformer.sh Autoformer_AAPL AAPL.csv 19
# sh scripts/Transformer.sh Transformer_AAPL AAPL.csv 19


# # # AAPL_pct
# sh scripts/PatchTST.sh PatchTST_AAPL_pct AAPL_pct 19
# sh scripts/DLinear.sh DLinear_AAPL_pct AAPL_pct 19
# sh scripts/Autoformer.sh Autoformer_AAPL_pct AAPL_pct 19
# sh scripts/Transformer.sh Transformer_AAPL_pct AAPL_pct 19

symbols=("JPM" "BAC" \
        "KO" "PG" \
        "JNJ" "PFE" \
        "XOM" "CVX" )

# other symbols
for symbol in "${symbols[@]}"
do
    sh scripts/PatchTST.sh PatchTST_$symbol $symbol.csv 19
    sh scripts/PatchTST.sh PatchTST_${symbol}_pct ${symbol}_pct.csv 19

done