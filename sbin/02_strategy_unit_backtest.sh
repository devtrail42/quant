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

# python strategy_unit_backtest/02_strategy_unit_backtest.py --interval ${target_interval} --target_strategy ${target_strategy} > ${root_dir}/var/log/strategy_unit_backtest/log_02_${target_strategy}.${target_interval}.txt 2>&1

python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_du > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_1.minute60.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_du > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_1.minute240.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_du_3 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_3.minute240.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_du_3 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_3.minute60.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_dru > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_dru.minute60.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_du_2 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_2.minute240.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_du_2 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_2.minute60.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_du_4 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_4.minute240.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_du_4 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_4.minute60.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_dru > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_dru.minute240.txt 2>&1 &
wait

python strategy_unit_backtest/02_strategy_unit_backtest.py --interval day --target_strategy low_bb_du > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_1.day.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval day --target_strategy low_bb_du_2 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_2.day.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval day --target_strategy low_bb_du_3 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_3.day.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval day --target_strategy low_bb_du_4 > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_du_4.day.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval day --target_strategy low_bb_dru > ${root_dir}/var/log/strategy_unit_backtest/log_02_low_bb_dru.day.txt 2>&1 &
wait


cd ${root_dir}/var/log/strategy_unit_backtest
python anal_result.py > anal.txt
cd ${root_dir}/sbin