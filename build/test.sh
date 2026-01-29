export CUDA_VISIBLE_DEVICES=7

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}/test"
pytest -ss test_esa_repre.py


