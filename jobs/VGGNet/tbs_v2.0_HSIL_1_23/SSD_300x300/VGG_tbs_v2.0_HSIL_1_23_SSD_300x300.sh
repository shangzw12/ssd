cd /home/shangzw/Desktop/SSD/caffe
./build/tools/caffe train \
--solver="models/VGGNet/tbs_v2.0_HSIL_1_23/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/tbs_v2.0_HSIL_1_23/SSD_300x300/VGG_tbs_v2.0_HSIL_1_23_SSD_300x300_1496816725_0.log
