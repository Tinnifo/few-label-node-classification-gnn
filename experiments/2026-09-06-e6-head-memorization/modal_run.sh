set -euo pipefail
python -c "import torch, sklearn, scipy; print('torch',torch.__version__,'threads',torch.get_num_threads())"
python fetch.py
mkdir -p out
python e5_probe_cka.py 2>&1 | tee out/e5_run.log
python e6_head_alone.py 2>&1 | tee out/e6_run.log
ls -la out/e5 out/e6
