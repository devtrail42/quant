export PATH=/Users/yongbeom/opt/miniconda3/bin:/Users/yongbeom/opt/miniconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /Users/yongbeom/opt/miniconda3/etc/profile.d/conda.sh
cd /Users/yongbeom/cyb/project/2025/quant/sbin
conda activate
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate stock
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

target_strategy=$1
target_interval=$2

market=coin
root_dir=$(cd .. && pwd)

if [ -z "$target_strategy" ]; then
    target_strategy=low_bb_du_2
fi

if [ -z "$target_interval" ]; then
    target_interval=minute60
fi

# python strategy_timeseries_backtest/04_strategy_timeseries_backtest.py --root_dir ${root_dir} --market ${market} --target_strategy ${target_strategy} --interval ${target_interval} > ${root_dir}/var/log/strategy_timeseries_backtest/${target_strategy}.${target_interval}.txt 2>&1
# python strategy_timeseries_backtest/04_strategy_timeseries_backtest.py --root_dir ${root_dir} --market ${market} --target_strategy low_bb_dru --interval minute60

mkdir -p ${root_dir}/var/log/strategy_timeseries_backtest
python strategy_timeseries_backtest/04_strategy_timeseries_backtest.py --root_dir ${root_dir} --market ${market} --target_strategy low_bb_dru --interval minute60 > ${root_dir}/var/log/strategy_timeseries_backtest/low_bb_dru.minute60.txt 2>&1 &
python strategy_timeseries_backtest/04_strategy_timeseries_backtest.py --root_dir ${root_dir} --market ${market} --target_strategy low_bb_dru --interval day > ${root_dir}/var/log/strategy_timeseries_backtest/low_bb_dru.day.txt 2>&1 &
python strategy_timeseries_backtest/04_strategy_timeseries_backtest.py --root_dir ${root_dir} --market ${market} --target_strategy low_bb_du_2 --interval minute60 > ${root_dir}/var/log/strategy_timeseries_backtest/low_bb_du_2.minute60.txt 2>&1 &
python strategy_timeseries_backtest/04_strategy_timeseries_backtest.py --root_dir ${root_dir} --market ${market} --target_strategy low_bb_du_2 --interval minute240 > ${root_dir}/var/log/strategy_timeseries_backtest/low_bb_du_2.minute240.txt 2>&1 &
python strategy_timeseries_backtest/04_strategy_timeseries_backtest.py --root_dir ${root_dir} --market ${market} --target_strategy low_bb_du_4 --interval minute60 > ${root_dir}/var/log/strategy_timeseries_backtest/low_bb_du_4.minute60.txt 2>&1 &
python strategy_timeseries_backtest/04_strategy_timeseries_backtest.py --root_dir ${root_dir} --market ${market} --target_strategy low_bb_du_4 --interval minute60 > ${root_dir}/var/log/strategy_timeseries_backtest/low_bb_du_4.minute240.txt 2>&1 &
wait