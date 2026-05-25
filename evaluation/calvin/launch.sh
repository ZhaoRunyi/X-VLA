#!/usr/bin/env bash
#SBATCH -p mozi_t
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --job-name=calvin
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# ---------- config ----------
NUM_WORKERS=10
CHUNK=100
BASE_PORT=8000
X_VLA_ENV=${X_VLA_ENV:-/path/to/X-VLA/.venv}
CALVIN_ENV=${CALVIN_ENV:-/path/to/calvin/.venv}

MODEL_PATH=/mnt/petrelfs/zhengjinliang/HeteroDiffusionPolicy/X-VLA-ckpt/news/X-VLA-Calvin-ABC_D
EXP_ROOT=exp/calvin/logs-0105-25-5
EVAL_DIR=evaluation/calvin
# ----------------------------

i=${SLURM_ARRAY_TASK_ID}

START=$((i * CHUNK))
END=$(((i + 1) * CHUNK))
PORT=$((BASE_PORT + i))


OUT_DIR=${EXP_ROOT}/worker_${i}
mkdir -p ${OUT_DIR}
OUT_DIR=$(realpath ${EXP_ROOT}/worker_${i})
INFO_JSON=${OUT_DIR}/info.json


exec > "${OUT_DIR}/slurm.out" 2>&1
echo "start server [worker ${i}] range=[${START},${END}) port=${PORT}"

# ---------- server ----------
source "${X_VLA_ENV}/bin/activate"
python -m deploy \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUT_DIR} \
    --port ${PORT} \
    --disable_slurm \
    > ${OUT_DIR}/server.log 2>&1 &

SERVER_PID=$!

echo "[worker ${i}] waiting for server ready"
while [ ! -f ${INFO_JSON} ]; do sleep 1; done
echo "[worker ${i}] server ready"

# ---------- client ----------
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
source "${CALVIN_ENV}/bin/activate"
cd ${EVAL_DIR}

python -u calvin_client.py \
    --output_dir ${OUT_DIR} \
    --server_ip 0.0.0.0 \
    --server_port ${PORT} \
    --eval_start ${START} \
    --eval_end ${END} \
    > "${OUT_DIR}/client.log" 2>&1

echo "[worker ${i}] client finished"

# ---------- cleanup ----------
kill ${SERVER_PID}
echo "[worker ${i}] done"
