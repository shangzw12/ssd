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
saved_file=$bash_dir/save.txt
test_name_size=$bash_dir/test_name_size
#all lines except first 100 are used to train
tmp_file_name=$bash_dir/tmp_file_name.txt
ls $img_dir | shuf > $tmp_file_name
cat $tmp_file_name | tail -n +270 | cut -d . -f 1 > $train_file_name
cat $train_file_name > $tmp_train_file_name
cat $tmp_file_name | head -n 270 | cut -d . -f 1 > $test_file_name
cat $test_file_name > $tmp_test_file_name
cat $test_file_name > $saved_file

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
# create test_name_size
sed -i "s/$/ 300 300/g" $saved_file
mv $saved_file $test_name_size

rm -f $tmp_train_file_name
rm -f $tmp_test_file_name
rm -f $tmp_file_name
rm -f $saved_file
