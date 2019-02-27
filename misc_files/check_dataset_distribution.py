# Visualize the DCASE 2019 SELD task dataset distribution

import os
import numpy as np
import sys
sys.path.append(os.path.join(sys.path[0], '..'))
import cls_feature_class
import matplotlib.pyplot as plot
plot.switch_backend('Qt4Agg')
# plot.switch_backend('TkAgg')
from IPython import embed
# Path to the metadata folder
dev_dataset = '/home/adavanne/taitoSharedData/DCASE2019/dataset/metadata_dev'

feat_cls = cls_feature_class.FeatureClass()
hop_len_s = feat_cls.get_hop_len_sec()
max_frames = feat_cls.get_nb_frames()
unique_classes_dict = feat_cls.get_classes()
nb_classes = len(unique_classes_dict)
azi_list, ele_list = feat_cls.get_azi_ele_list()
min_azi_ind = min(azi_list)//10
min_ele_ind = min(ele_list)//10
nb_ir = 5
nb_files_per_split = [0]*5
split_info_dic = {}
for dataset_path in [dev_dataset]:
    for file in os.listdir(dataset_path):
        desc_dict = feat_cls.read_desc_file(os.path.join(dataset_path, file))
        split = int(file[5])
        ir = int(file[9])
        ov = int(file[13])
        nb_files_per_split[split] += 1
        if split not in split_info_dic:
            split_info_dic[split] = {
                'scop': np.zeros(nb_classes),
                'length': np.zeros(nb_classes),
                'se_cnt': np.zeros(nb_classes),
                'azi_ir': np.zeros((nb_ir, len(azi_list))),
                'ele_ir': np.zeros((nb_ir, len(ele_list))),
                'azi_ele_ir': np.zeros((nb_ir, len(azi_list)*len(ele_list))),

            }

        labels = feat_cls.get_clas_labels_for_file(desc_dict)

        # Number of frames in which same class overlaps
        split_info_dic[split]['scop'] += np.sum(labels.sum(2)==2, 0)

        for i, se_class in enumerate(desc_dict['class']):
            se_class = unique_classes_dict[se_class]
            start = desc_dict['start'][i]
            end = desc_dict['end'][i]
            split_info_dic[split]['length'][se_class] += (end - start) * hop_len_s
            split_info_dic[split]['se_cnt'][se_class] += 1

            azi = desc_dict['azi'][i]
            ele = desc_dict['ele'][i]

            split_info_dic[split]['azi_ir'][ir, (azi // 10) - min_azi_ind] += 1
            split_info_dic[split]['ele_ir'][ir, (ele // 10) - min_ele_ind] += 1

            split_info_dic[split]['azi_ele_ir'][ir, feat_cls.get_list_index(azi, ele)] += 1

cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
azi_list = np.array(azi_list)
ele_list = np.array(ele_list)
azi_ele_list = np.arange(len(azi_list)*len(ele_list))
nb_split = len(split_info_dic.keys())


split_list = np.sort(list(split_info_dic))
plot.figure()
plot.subplot(211)
plot.title('Same class overlap percentage')
scop_list = []
for i in split_list:
    scop_list.append((100.*np.sum(split_info_dic[i]['scop']))/(nb_files_per_split[i]*max_frames))
    plot.bar(np.arange(nb_classes) + 0.1 * i, (100. * split_info_dic[i]['scop']) / (nb_files_per_split[i] * max_frames),
             color=cmap[i], width=.15, label='split {}'.format(i))
plot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=nb_split, mode="expand", borderaxespad=0.)
plot.ylabel('SCOP')
plot.xlabel('Class index')

plot.subplot(212)
plot.bar(split_list, scop_list)
plot.xlabel('Split')
plot.ylabel('SCOP')
plot.title('SCOP average across dataset: {}%'.format(np.mean(scop_list)))

plot.figure()
plot.title('Total length of classes')
for i in split_list:
    plot.bar(np.arange(nb_classes)+0.1*i, split_info_dic[i]['length'], color=cmap[i], width=.15, label='split {}'.format(i))
plot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=nb_split, mode="expand", borderaxespad=0.)
plot.ylabel('length in sec')
plot.xlabel('Class index')


plot.figure()
plot.title('Number of examples per class')
for i in np.sort(list(split_info_dic)):
    plot.bar(np.arange(nb_classes)+0.1*i, split_info_dic[i]['se_cnt'], color=cmap[i], width=.15, label='split {}'.format(i))
plot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=nb_split, mode="expand", borderaxespad=0.)
plot.ylabel('number of examples/class')
plot.xlabel('Class index')

plot.figure()
for i in split_list:
    for j in range(nb_ir):
        plot.subplot(nb_split, 1, i)
        plot.title('Azimuth distribution per impulse response in Split {}'.format(i))
        plot.bar(azi_list+j, split_info_dic[i]['azi_ir'][j], color=cmap[j], width=1.5, label='ir {}'.format(j) if (i ==1) else None)
    if i == 1:
        plot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=nb_ir, mode="expand", borderaxespad=0.)
    plot.ylabel('Frequency')
plot.xlabel('Azimuth angles')

plot.figure()
for i in split_list:
    for j in range(nb_ir):
        plot.subplot(nb_split, 1, i)
        plot.title('Elevation distribution per impulse response in Split {}'.format(i))
        plot.bar(ele_list+j, split_info_dic[i]['ele_ir'][j], color=cmap[j], width=1.5, label='ir {}'.format(j) if (i ==1) else None)
    if i == 1:
        plot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=nb_ir, mode="expand", borderaxespad=0.)
    plot.ylabel('Frequency')
plot.xlabel('Elevation angles')

plot.figure()
for i in split_list:
    for j in range(nb_ir):
        plot.subplot(nb_split, 1, i)
        plot.title('Azimuth-Elevation combination distribution per impulse response in Split {}'.format(i))
        plot.bar(azi_ele_list+.1*j, split_info_dic[i]['azi_ele_ir'][j], color=cmap[j], width=.15, label='ir {}'.format(j) if (i ==1) else None)
    if i == 1:
        plot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=nb_ir, mode="expand", borderaxespad=0.)

    plot.ylabel('Frequency')
plot.xlabel('Azimuth*Elevation angles')
plot.show()
