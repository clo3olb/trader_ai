# model_name, data_name, input_size

set -e

# AAPL
# sh scripts/DLinear.sh DLinear_AAPL AAPL.csv 19
# sh scripts/Autoformer.sh Autoformer_AAPL AAPL.csv 18
# sh scripts/Transformer.sh Transformer_AAPL AAPL.csv 18

# # AAPL_pct
# sh scripts/DLinear.sh DLinear_AAPL_pct AAPL_pct.csv 19
sh scripts/Autoformer.sh Autoformer_AAPL_pct AAPL_pct.csv 18
sh scripts/Transformer.sh Transformer_AAPL_pct AAPL_pct.csv 18

symbols=("PG" \
        "JNJ" "PFE" \
        "XOM" "CVX" )

# other symbols
for symbol in "${symbols[@]}"
do
    sh scripts/PatchTST.sh PatchTST_$symbol $symbol.csv 19
    # sh scripts/PatchTST.sh PatchTST_${symbol}_pct ${symbol}_pct.csv 19

done