export TRANSFORMERS_OFFLINE=1
export IPEX_COMPUTE_ENG=1
export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
python qinling_all.py --device xpu --precision fp32 2>&1 | tee torchbench_fp32.log
python qinling_all.py --device xpu --precision amp 2>&1 | tee torchbench_amp.log
python qinling_all.py --device xpu --precision amp --optimize True 2>&1 | tee torchbench_amp_optimize.log
python qinling_all.py --device xpu --precision amp --optimize True --jit True 2>&1 | tee torchbench_amp_optimize_jit.log
python qinling_all.py --device xpu --precision amp --jit True 2>&1 | tee torchbench_amp_jit.log
python qinling_all.py --device xpu --precision fp32 --jit True 2>&1 | tee torchbench_fp32_jit.log
