#!/bin/bash

root_dir=$HOME/data/tbs
sub_dir=tbs_300x300_HSIL_10x
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

img_dir=$root_dir/$sub_dir/Images
train_file_name=$bash_dir/trainval.txt
tmp_train_file_name=$bash_dir/tmp_trainval.txt
test_file_name=$bash_dir/test.txt
tmp_test_file_name=$bash_dir/tmp_test.txt
tmp_file=$bash_dir/tmp.txt
#all lines except first 100 are used to train
tmp_file_name=$bash_dir/tmp_file_name.txt
ls $img_dir | sort > $tmp_file_name
cat $tmp_file_name | tail -n +101 | cut -d . -f 1 > $train_file_name
cat $train_file_name > $tmp_train_file_name
cat $tmp_file_name | head -n 100 | cut -d . -f 1 > $test_file_name
cat $test_file_name > $tmp_test_file_name

#append dir name in front of file name(each line)

sed -i "s/^/$sub_dir\/Images\//g" $train_file_name
sed -i "s/$/.jpg/g" $train_file_name
sed -i "s/^/$sub_dir\/Annotations\//g" $tmp_train_file_name
sed -i "s/$/.xml/g" $tmp_train_file_name
paste -d' ' $train_file_name $tmp_train_file_name >> $tmp_file
mv $tmp_file $train_file_name

sed -i "s/^/$sub_dir\/Images\//g" $test_file_name
sed -i "s/$/.jpg/g" $test_file_name
sed -i "s/^/$sub_dir\/Annotations\//g" $tmp_test_file_name
sed -i "s/$/.xml/g" $tmp_test_file_name
paste -d' ' $test_file_name $tmp_test_file_name >> $tmp_file
mv $tmp_file $test_file_name

rm -f $tmp_train_file_name
rm -f $tmp_test_file_name
rm -f $tmp_file_name
