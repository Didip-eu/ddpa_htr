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

img_files=$*

echo APPDIR=$APPDIR

# check that segmentation file is there

for img in $img_files; do

	segmentation_file=${img%.img.jpg}.lines.pred.xml

	echo $segmentation_file
	if [[ ! -f $segmentation_file ]]; then
		echo "No segmentation file found for file ${img} → running segmentation app."
		(cd ${APPDIR}/ddpa_lines;
		PYTHONPATH=${APPDIR}/ddpa_lines bin/ddp_line_detect -img_paths $img;
		cd ${ROOT})
	fi

	# run htr
	if [[ -f $segmentation_file ]]; then
		(cd ${APPDIR}/ddpa_htr; 
		PYTHONPATH=${APPDIR}/ddpa_htr bin/ddp_htr_inference -img_paths $img;
		cd ${ROOT})
	fi
done

