#!/bin/bash

declare -a datasets=("segment" "satimage" "pendigits")

for data in "${datasets[@]}"
do
	python fit_folds.py "$data" >> r2svm_medium.log
done
