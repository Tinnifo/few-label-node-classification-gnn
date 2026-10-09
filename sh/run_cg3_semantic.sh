#!/usr/bin/bash -l
#SBATCH --job-name=CG3SEM
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition batch
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00

nvidia-smi

# Global variables
BASEFOLDER=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=python
SCRIPT_PATH=${BASEFOLDER}/src/cg3_semantic.py

# Experiment parameters
DATASET=cora
LABEL_STRATEGY=per_class   # per_class | percentage
BUDGET=20                  # labels per class, or a fraction of all nodes for percentage
SEEDS=0,1,2,3,4
LOCAL_MODEL=gcn            # gcn | gat
GLOBAL_MODEL=hgcn          # hgcn | hgat
EPOCHS=200
LR=0.01
WEIGHT_DECAY=5e-4
DROPOUT=0.6
HIDDEN_LOCAL=1024
HIDDEN_GLOBAL=32
COARSEN_LEVEL=4
CHANNEL_NUM=4
SEMANTIC=""                # "--semantic --texts ${BASEFOLDER}/data/${DATASET}_texts.txt" turns the semantic view on

# Output
MODELS_DIR=${BASEFOLDER}/models
mkdir -p ${MODELS_DIR}

# Command
CMD="${PYTHON} ${SCRIPT_PATH} --dataset ${DATASET} --label-strategy ${LABEL_STRATEGY} --budget ${BUDGET} --seeds ${SEEDS}"
CMD="${CMD} --local-model ${LOCAL_MODEL} --global-model ${GLOBAL_MODEL} --epochs ${EPOCHS} --lr ${LR}"
CMD="${CMD} --weight-decay ${WEIGHT_DECAY} --dropout ${DROPOUT} --hidden-local ${HIDDEN_LOCAL} --hidden-global ${HIDDEN_GLOBAL}"
CMD="${CMD} --coarsen-level ${COARSEN_LEVEL} --channel-num ${CHANNEL_NUM} --early-stopping --output ${MODELS_DIR} ${SEMANTIC}"

echo ${CMD}
${CMD}
