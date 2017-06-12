import numpy as np
import matplotlib.pyplot as plt
import os
caffe_root = os.environ['CAFFE_ROOT']
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'data/tbs_v2.0_HSIL_1_23/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

model_def = 'models/VGGNet/tbs_v2.0_HSIL_1_23/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/tbs_v2.0_HSIL_1_23/SSD_300x300/VGG_tbs_v2.0_HSIL_1_23_SSD_300x300_iter_200000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

##SSD detection
def one_image_detection(url):
    image_resize = 300
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    image = caffe.io.load_image(images_dir + url)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    # Forward pass.
    detections = net.forward()['detection_out']
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.clf()
    plt.imshow(image)
    currentAxis = plt.gca()

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    #plt.show()
    plt.savefig(results_dir + "/" + url)

test_name_size = caffe_root + "/data/tbs_v2.0_HSIL_1_23/test_name_size"
images_dir = os.environ['HOME'] + "/data/tbs_v2.0_HSIL_1_23/Images/"
results_dir = os.environ['HOME'] + "/data/tbs_v2.0_HSIL_1_23/results/SSD_300x300/Main/"
fn_results_dir = os.environ['HOME']+ "/data/tbs_v2.0_HSIL_1_23/results/SSD_300x300/fn_result/"

#get all gts, and return back as {'HSIL-1': [], 'HSIL-2': [], 'HSIL-3': []}
def get_gts(url):
    #url is sth like xxx.jpg
    annots_dir = '/home/deepcare/data/tbs_v2.0_HSIL_1_23/Annotations/'
    url = url.split('.')[0] + '.xml'
    file_name = annots_dir + url
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_name)
    root = tree.getroot()
    res = {'HSIL-1': [], 'HSIL-2': [], 'HSIL-3': []}
    for child in root:
        if child.tag == 'object':
            res[child[0].text].append((int(child[1][0].text)
                                        ,int(child[1][1].text)
                                        ,int(child[1][2].text)
                                        ,int(child[1][3].text)
                                        ))
    return res

def one_image_detection_with_gts(url, only_fn):
    image_resize = 300
    image_dir = images_dir +  url
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    image = caffe.io.load_image(image_dir)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    # Forward pass.
    detections = net.forward()['detection_out']
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.clf()
    plt.imshow(image)
    currentAxis = plt.gca()

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    #plt.show()
    gts = get_gts(url)
    is_detected = True
    for key in gts:
    	is_detected_one = False
        vals = gts[key]
        for i in range(len(vals)):
            label_name = key
            x_min, y_min, x_max, y_max = vals[i]
            if only_fn:
            	for i in xrange(top_conf.shape[0]):
        			xmin = int(round(top_xmin[i] * image.shape[1]))
        			ymin = int(round(top_ymin[i] * image.shape[0]))
        			xmax = int(round(top_xmax[i] * image.shape[1]))
        			ymax = int(round(top_ymax[i] * image.shape[0]))
        			if overlap((xmin, ymin, xmax, ymax), (x_min, y_min, x_max, y_max)) > 0.000000001:
        				is_detected_one = True
                if not is_detected_one:
        			is_detected = False

            coords = (x_min, y_min), x_max-x_min +1, y_max - y_min +1
            color = 'green'
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(x_min, y_min, label_name, bbox={'facecolor':color, 'alpha':0.5})
    if not only_fn:
    	plt.savefig(results_dir + "/" + url)
    else:
    	if not is_detected:
    		plt.savefig(fn_results_dir + "/" + url)


def gen_results():
    with open(test_name_size) as rd:
        reader = rd.read()
    reader = reader.split('\n')
    reader = reader[0: len(reader) - 2]
    for i in range(len(reader)):
        line = reader[i]
        line = line.split(' ')
        file_name = line[0] + ".jpg"
        one_image_detection(file_name)

def gen_results_with_gts(only_fn):
    with open(test_name_size) as rd:
        reader = rd.read()
    reader = reader.split('\n')
    reader = reader[0: len(reader) - 2]
    for i in range(len(reader)):
        line = reader[i]
        line = line.split(' ')
        file_name = line[0] + ".jpg"
        one_image_detection_with_gts(file_name, only_fn)

def overlap(coord_0, coord_1):
    xmin,  ymin, xmax, ymax = coord_0
    x_min, y_min, x_max, y_max = coord_1
    whole_area = (xmax-xmin) * (ymax- ymin) + 0.0
    if x_min <= xmin and y_min <= ymin:
        #left_top
        width = min(x_max, xmax) - xmin
        height = min(y_max, ymax) - ymin
        if width <0  or height <0:
            return 0
        else:
            return width * height / whole_area
    elif y_min <= ymin:
        #top right
        width = min(x_max, xmax) - x_min
        height = min(y_max, ymax) - ymin
        if width <0  or height <0:
            return 0
        else:
            return width * height / whole_area
    elif x_min <= xmin:
        #bottom left
        width = min(x_max, xmax) - xmin
        height = min(y_max, ymax) - y_min
        if width <0  or height <0:
            return 0
        else:
            return width * height / whole_area
    else:
        #bottom right
        width = min(x_max, xmax) - x_min
        height = min(y_max, ymax) - y_min
        if width <0  or height <0:
            return 0
        else:
            return width * height / whole_area

def one_image_false_negative_positive(url, ratio, _conf = 0.6):
    image_resize = 300
    image_dir = images_dir +  url
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    image = caffe.io.load_image(image_dir)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    # Forward pass.
    detections = net.forward()['detection_out']
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= _conf]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    currentAxis = plt.gca()
    gts = get_gts(url)
    false_negative = 0
    false_positive = 0
    is_fn = True
    is_fp = True
    all_gts = []
    for k in gts:
        all_gts.extend(gts[k])
    for k in range(len(all_gts)):
        is_fn = True
        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            #if label_name is in {'HSIL-1', 'HSIL-2', 'HSIL-3'}:
            if True:
                if overlap((xmin, ymin, xmax, ymax), all_gts[k]) > ratio:
                    is_fn = False
        if is_fn:
            false_negative += 1
    for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            #if label_name is in {'HSIL-1', 'HSIL-2', 'HSIL-3'}:
            if True:
                is_fp = True
                for k in range(len(all_gts)):
                    if overlap((xmin, ymin, xmax, ymax), all_gts[k]) > ratio:
                        is_fp = False
                if is_fp:
                    false_positive += 1

    return false_negative, false_positive, len(all_gts)

def get_false_negative_positive(ratio = 0.5, conf_ratio=0.6):
    fn_num = 0
    gt_num = 0
    fp_num = 0
    with open(test_name_size) as rd:
        reader = rd.read()
    reader = reader.split('\n')
    reader = reader[0: len(reader) - 2]
    for i in range(len(reader)):
        line = reader[i]
        line = line.split(' ')
        file_name = line[0] + ".jpg"
        fn, fp, gt= one_image_false_negative_positive(file_name, ratio, conf_ratio)
        gt_num += gt
        fn_num += fn
        fp_num += fp
    return fn_num , fp_num , gt_num
def plot(x_axis, fns, fps, gts):#all are ratios
    x_fn = range(len(fns))
    x_fp = range(len(fps))

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2, right=0.85)

    newax = fig.add_axes(ax.get_position())
    newax.patch.set_visible(False)

    newax.yaxis.set_label_position('right')
    newax.yaxis.set_ticks_position('right')

    newax.spines['bottom'].set_position(('outward', 35))

    ax.plot(x_fn, fns, 'r-')
    ax.set_xlabel('X_fn', color='red')
    ax.set_ylabel('Loss', color='red')

    x = np.linspace(0, 6*np.pi)
    newax.plot(x_fp, fps , 'g-')

    newax.set_xlabel('X_fp', color='green')
    newax.set_ylabel('Detection_eval', color='green')
    plt.show()


def solve():
    #gen_results_with_gts()
    overlap_ratio = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5] #7
    conf_ratio = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] #9
    #overlap_ratio = [0.5]
    #conf_ratio = [0.6]
    fns = []
    fps = []
    gt = 0
    x_axis = []
    for i in range(len(overlap_ratio)):
        for j in range(len(conf_ratio)):
            fn_num, fp_num, gt_num  = get_false_negative_positive(overlap_ratio[i], conf_ratio[j])
            print overlap_ratio[i],\
                conf_ratio[j], 1.0 - (fn_num+0.0)/ gt_num, 1.0 - (fp_num+0.0)/gt_num
            x_axis.append((overlap_ratio[i], conf_ratio[j]))
            fns.append(1.0 - (fn_num+0.0)/ gt_num)
            fps.append(1.0 - (fp_num+0.0)/gt_num)
            gt = gt_num
    plot(x_axis, fns, fps, gt)


solve()
#gen_results_with_gts(True)
