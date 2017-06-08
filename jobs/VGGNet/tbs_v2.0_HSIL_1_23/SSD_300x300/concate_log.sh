prefix='VGG_tbs_300x300_HSIL_10x_SSD_300x300_'
head=$1
tmp_file_name='tmp_file_name.txt'
tmp_fname_2='tmp_file_name_2.txt'
cont_fname='cont_file.txt'
sub_dir='./plot'
mbox_loss='mbox_loss.txt'
detections_eval='detections_eval.txt'
#contacate files from head until the end
ls $PWD | sort | grep "$1" -A 100 > $tmp_file_name
cat $tmp_file_name | grep -v "sh" > $tmp_fname_2
cat $(ls $(cat $tmp_fname_2)) > $cont_fname
rm -r $tmp_file_name
rm -r $tmp_fname_2
cat $cont_fname | grep "mbox_loss" | grep "*"| grep -v "nan" > $sub_dir/$mbox_loss
cat $cont_fname | grep "detection_eval"| grep -v "nan" | grep "=" > $sub_dir/$detections_eval
rm -r $cont_fname
#cd $sub_dir 
python './plot/plot.py'
