export CUDA_VISIBLE_DEVICES=3

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}/test"
pytest -ss test_vec_add.py


