#!/bin/bash

usage() {
    echo "Usage: $0 -v <AMR corpus version. Possible values: 1 or 2> -p <Path to AMR corpus>"
    echo "  Make sure your AMR corpus is untouched."
    echo "  It should organized like below:"
    echo "  <AMR corpus>"
    echo "      data/"
    echo "      docs/"
    echo "      index.html"
    exit 1;
}

while getopts ":h:v:p:" o; do
    case "${o}" in
        h)
            usage
            ;;
        v)
            v=${OPTARG}
            ((v == 1 || v == 2)) || usage
            ;;
        p)
            p=${OPTARG}
            ;;
        \? )
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z $v ]; then
    usage
fi

if [ -z $p ]; then
    usage
fi


if [[ "$v" == "2" ]]; then
    DATA_DIR=data/AMR/amr_2.0
    SPLIT_DIR=$p/data/amrs/split
    TRAIN=${SPLIT_DIR}/training
    DEV=${SPLIT_DIR}/dev
    TEST=${SPLIT_DIR}/test
else
    DATA_DIR=data/AMR/amr_1.0
    SPLIT_DIR=$p/data/amrs/split
    TRAIN=${SPLIT_DIR}/training
    DEV=${SPLIT_DIR}/dev
    TEST=${SPLIT_DIR}/test
fi

echo "Preparing data in ${DATA_DIR}...`date`"
mkdir -p ${DATA_DIR}
awk FNR!=1 ${TRAIN}/* > ${DATA_DIR}/train.txt
awk FNR!=1 ${DEV}/* > ${DATA_DIR}/dev.txt
awk FNR!=1 ${TEST}/* > ${DATA_DIR}/test.txt
echo "Done..`date`"

