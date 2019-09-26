#!/bin/bash

# 本脚本创建imagenet数据集的子集软链接
mkdir -p /data/imagenetsubset/train /data/imagenetsubset/val

train_labels=(/data/imagenet/train/*)
n_classes=${#train_labels[@]}
echo $n_classes
n_classes_sub=16
train_labels_subset=("${train_labels[@]:1:$n_classes_sub}")
labels_subset=()
for i in "${train_labels_subset[@]}"
do
	label=${i##*/}
	echo $label
	labels_subset+=($label)
	trainlndst="/data/imagenetsubset/train/"$label
	vallndst="/data/imagenetsubset/val/"$label
	trainlnsrc="/data/imagenet/train/"$label
	vallnsrc="/data/imagenet/val/"$label
	ln -s $trainlnsrc $trainlndst
	ln -s $vallnsrc $vallndst
done

