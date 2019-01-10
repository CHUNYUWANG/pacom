#!/bin/bash

set -x
DATA_DIR=$PWD
LOG_DIR=$PWD/log
MODEL_DIR=$PWD/output
CFG=$PWD/experiments/mixed/resnet50/256_fusion.yaml

# parsing command line arguments:
while [[ $# > 0 ]]
do
key="$1"

case $key in
    -h|--help)
    echo "Usage: toolkit-execute [run_options]"
    echo "Options:"
    echo "  -c|--cfg <config> - which configuration file to use (default NONE)"
    echo "  -d|--dataDir <path> - directory path to input files (default NONE)"
    echo "  -l|--logDir <path> - directory path to save the log files (default \$PWD)"
    echo "  -m|--modelDir <path> - directory path to save the model files (default NONE)"
    exit 1
    ;;
    -c|--cfg)
    CFG="$2"
    shift # pass argument
    ;;
    -d|--dataDir)
    DATA_DIR="$2"
    shift # pass argument
    ;;
    -l|--logDir)
    LOG_DIR="$2"
    shift # pass argument
    ;;
    -m|--modelDir)
    MODEL_DIR="$2"
    shift # pass argument
    ;;
    *)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    ;;
esac
shift # past argument or value
done

echo "train"
python run/pose2d/train.py --cfg ${CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} ${EXTRA_ARGS}

echo "test"
python run/pose2d/valid.py --cfg ${CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} ${EXTRA_ARGS} --state final


