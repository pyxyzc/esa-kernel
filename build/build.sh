PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "${PROJECT_DIR}/build"
rm -rf *.so

cd "${PROJECT_DIR}/test"
rm -rf *.so

cd "${PROJECT_DIR}/build"
python util.py
mv *.so "${PROJECT_DIR}/test"

