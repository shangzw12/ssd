cd /home/deepcare/Desktop/SSD/caffe
./build/tools/caffe train \
--solver="models/VGGNet/tbs_v2.0_HSIL_1_23/SSD_300x300/solver.prototxt" \
--snapshot="models/VGGNet/tbs_v2.0_HSIL_1_23/SSD_300x300/VGG_tbs_v2.0_HSIL_1_23_SSD_300x300_iter_5095.solverstate" \
--gpu 0 2>&1 | tee jobs/VGGNet/tbs_v2.0_HSIL_1_23/SSD_300x300/VGG_tbs_v2.0_HSIL_1_23_SSD_300x300_1496829047_5095.log
