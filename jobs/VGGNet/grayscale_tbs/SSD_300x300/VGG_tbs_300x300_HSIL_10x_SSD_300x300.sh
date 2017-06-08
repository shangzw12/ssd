cd /home/shangzw/Desktop/SSD/caffe
./build/tools/caffe train \
--solver="models/VGGNet/grayscale_tbs/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/grayscale_tbs/SSD_300x300/VGG_tbs_300x300_HSIL_10x_SSD_300x300_1496740193_0.log
