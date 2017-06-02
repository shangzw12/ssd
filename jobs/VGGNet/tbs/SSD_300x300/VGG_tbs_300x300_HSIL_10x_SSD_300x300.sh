cd /home/shangzw/Desktop/SSD/caffe
./build/tools/caffe train \
--solver="models/VGGNet/tbs/SSD_300x300/solver.prototxt" \
--snapshot="models/VGGNet/tbs/SSD_300x300/VGG_tbs_300x300_HSIL_10x_SSD_300x300_iter_18000.solverstate" \
--gpu 0 2>&1 | tee jobs/VGGNet/tbs/SSD_300x300/VGG_tbs_300x300_HSIL_10x_SSD_300x300_1496385562_18000.log
