export PATH=/Users/yongbeom/opt/miniconda3/bin:/Users/yongbeom/opt/miniconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /Users/yongbeom/opt/miniconda3/etc/profile.d/conda.sh
cd /Users/yongbeom/cyb/project/2025/quant/sbin
conda activate
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate stock
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

target_strategy_feature=$1
target_interval=$2
market=coin
root_dir=$(cd .. && pwd)

if [ -z "$target_strategy" ]; then
    target_strategy=low_bb_du
fi

if [ -z "$target_interval" ]; then
    target_interval=minute60
fi

python train_xgb/05_get_strategy_filtered_data.py --root_dir ${root_dir} --market ${market} --target_strategy_feature low_bb_du --interval minute60
