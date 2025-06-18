#!/bin/sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
N='\033[0m' # No Color

echo "${GREEN}Building onnx2tf${N}"
python setup_custom.py sdist bdist_wheel
echo "${GREEN}Installing onnx2tf${N}"
pip install dist/onnx2tf-*.whl --force-reinstall
echo "${GREEN}Checking onnx2tf${N}"
pip show onnx2tf
echo "${GREEN}All steps completed successfully!${N}"
