#!/bin/sh -x
caffe_root=$CAFFE_ROOT
head=$1
model_name=$2
prefix='VGG_'$model_name'_SSD_300x300_'
log_dir='jobs/VGGNet/'$model_name'/SSD_300x300'
python_dir='python'
tmp_file_name='tmp_file_name_1.txt'
tmp_fname_2='tmp_file_name_2.txt'
cont_fname='cont_file.txt'
sub_dir='plot'
mbox_loss=$log_dir/$sub_dir/'mbox_loss.txt'
detections_eval=$log_dir/$sub_dir/'detections_eval.txt'
#contacate files from head until the end(max 100)
ls $log_dir | sort | grep "$1" -A 100 > $log_dir/$tmp_file_name #A stands for after
cat $log_dir/$tmp_file_name | grep -v "\.sh" > $log_dir/$tmp_fname_2
echo $(cat $log_dir/$tmp_file_name)
echo $(cat $log_dir/$tmp_fname_2)
cd $log_dir
cat $(ls $(cat $tmp_fname_2)) > $cont_fname
echo $cont_fname
rm -r $tmp_file_name
rm -r $tmp_fname_2
cat $cont_fname | grep "mbox_loss" | grep "*"| grep -v "nan" > $caffe_root/$mbox_loss
cat $cont_fname | grep "detection_eval"| grep -v "nan" | grep "=" > $caffe_root/$detections_eval
rm -r $cont_fname
cd $caffe_root

echo $PWD
python $python_dir/'plot.py' $log_dir/$sub_dir
