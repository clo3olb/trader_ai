# model_name, data_name, input_size

set -e

# AAPL
# sh scripts/DLinear.sh DLinear_AAPL AAPL.csv 19
# sh scripts/Autoformer.sh Autoformer_AAPL AAPL.csv 18
# sh scripts/Transformer.sh Transformer_AAPL AAPL.csv 18

# # AAPL_pct
# sh scripts/DLinear.sh DLinear_AAPL_pct AAPL_pct.csv 19
# sh scripts/Autoformer.sh Autoformer_AAPL_pct AAPL_pct.csv 18
# sh scripts/Transformer.sh Transformer_AAPL_pct AAPL_pct.csv 18

# symbols=("AAPL" "BAC" "CVX" "JNJ" "KO" "MSFT" "PFE" "PG" "XOM")

# sh scripts/PatchTST.sh PatchTST_AAPL_without_sentiment AAPL_without_sentiment.csv 19
sh scripts/PatchTST.sh PatchTST_AAPL_with_sentiment AAPL_with_sentiment.csv 20

sh scripts/PatchTST.sh PatchTST_BAC_without_sentiment BAC_without_sentiment.csv 19
sh scripts/PatchTST.sh PatchTST_BAC_with_sentiment BAC_with_sentiment.csv 20

sh scripts/PatchTST.sh PatchTST_CVX_without_sentiment CVX_without_sentiment.csv 19
sh scripts/PatchTST.sh PatchTST_CVX_with_sentiment CVX_with_sentiment.csv 20

sh scripts/PatchTST.sh PatchTST_JNJ_without_sentiment JNJ_without_sentiment.csv 19
sh scripts/PatchTST.sh PatchTST_JNJ_with_sentiment JNJ_with_sentiment.csv 20