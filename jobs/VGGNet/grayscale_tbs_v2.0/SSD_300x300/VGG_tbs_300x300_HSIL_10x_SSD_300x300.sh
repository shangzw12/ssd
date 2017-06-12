cd /home/ubuntu/Desktop/SSD/caffe
./build/tools/caffe train \
--solver="models/VGGNet/grayscale_tbs_v2.0/SSD_300x300/solver.prototxt" \
--snapshot="models/VGGNet/grayscale_tbs_v2.0/SSD_300x300/VGG_tbs_300x300_HSIL_10x_SSD_300x300_iter_37.solverstate" \
--gpu 0 2>&1 | tee jobs/VGGNet/grayscale_tbs_v2.0/SSD_300x300/VGG_tbs_300x300_HSIL_10x_SSD_300x300_1497000863_37.log
