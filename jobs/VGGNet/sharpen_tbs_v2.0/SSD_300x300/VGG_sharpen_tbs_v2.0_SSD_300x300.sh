cd /home/shangzw/Desktop/SSD/caffe
./build/tools/caffe train \
--solver="models/VGGNet/sharpen_tbs_v2.0/SSD_300x300/solver.prototxt" \
--snapshot="models/VGGNet/sharpen_tbs_v2.0/SSD_300x300/VGG_sharpen_tbs_v2.0_SSD_300x300_iter_75042.solverstate" \
--gpu 0 2>&1 | tee jobs/VGGNet/sharpen_tbs_v2.0/SSD_300x300/VGG_sharpen_tbs_v2.0_SSD_300x300_1497001808_75042.log
