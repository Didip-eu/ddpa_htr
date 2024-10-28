#!/usr/bin/bash

# htr_script.sh
# nprenet@gmail.com
# 10/2024
# A wrapper script for HTR inference on charters.

USAGE_STRING="USAGE: $0 img_file [ img_file2 ... ]"


if [[ $# -eq 0 ]] ; then
	echo ${USAGE_STRING};
	exit
fi

set -x
ROOT=$(dirname $(realpath "$0"))
echo ROOT=$ROOT
APPDIR=$(realpath ${ROOT}/..)




rewritten_target=$(echo "${@/.img.jpg/.htr.pred.json}")

make $rewritten_target
