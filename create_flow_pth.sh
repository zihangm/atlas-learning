#!/bin/bash


for from in 0 1 2
do
for to in 0 1 2 3 4
do
if [ $from -lt $to ]
then

arr=()
arr_inv=()
for i in $(seq 0 455)
do 
for j in $(seq 0 31)
do
if [ $i -lt 10 ]
then 
m="0"$i 
else
m=$i
fi
if [ $j -lt 10 ] 
then 
n="0"$j
else
n=$j
fi
#output=$( ls patches_dataset/0_to_1/0_"$m"_"$n"_1 | wc -l  )
#if [ -e "patches_dataset/"$from"_to_"$to"/"$from"_"$m"_"$n"_"$to"/"$from"_"$m"_"$n"_"$to"_inversewarp.nii" ]
#then
arr+=("../dataset/patches_dataset/"$from"_to_"$to"/"$from"_"$m"_"$n"_"$to"/"$from"_"$m"_"$n"_"$to"_inversewarp.nii")
#echo $(($i*32+$j)) >> 'training_dataset/mask/'$from"_to_"$to".txt"
#fi

#if [ -e "patches_dataset/"$from"_to_"$to"/"$from"_"$m"_"$n"_"$to"/"$from"_"$m"_"$n"_"$to"_warp.nii" ]
#then
arr_inv+=("../dataset/patches_dataset/"$from"_to_"$to"/"$from"_"$m"_"$n"_"$to"/"$from"_"$m"_"$n"_"$to"_warp.nii")
#echo $(($i*32+$j)) >> 'training_dataset/mask/'$to"_to_"$from".txt"
#fi
done
done
echo $from $to
python2 ./create_pth.py --files ${arr[@]} --output ../dataset/training_dataset'/'$from'_'to_$to'.'pth.tar
python2 ./create_pth.py --files ${arr_inv[@]} --output ../dataset/training_dataset'/'$to'_'to_$from'.'pth.tar

fi
done
done

