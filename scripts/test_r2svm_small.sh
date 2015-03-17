#!/bin/bash

declare -a datasets=("glass" "iris" "liver" "heart" "wine")

for data in "${datasets[@]}"
do
	python fit_folds.py "$data" >> r2svm_small.log
done
