#
# Data generator for training the SELDnet
#

import os
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random


class DataGenerator(object):
    def __init__(
            self, dataset='foa', feat_label_dir='', is_eval=False, split=1, batch_size=32, seq_len=64,
            shuffle=True, per_file=False
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(feat_label_dir=feat_label_dir, dataset=dataset, is_eval=is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._feat_len = None
        self._2_nb_ch = 2 * self._feat_cls.get_nb_channels()
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._class_dict = self._feat_cls.get_classes()
        self._nb_classes = len(self._class_dict.keys())
        self._default_azi, self._default_ele = self._feat_cls.get_default_azi_ele_regr()
        self._get_filenames_list_and_feat_label_sizes()

        self._batch_seq_len = self._batch_size*self._seq_len
        self._circ_buf_feat = None
        self._circ_buf_label = None

        if self._per_file:
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                               float(self._seq_len * self._batch_size))))

        # self._dummy_feat_vec = np.ones(self._feat_len.shape) *

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._feat_len, self._2_nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, seq_len: {}, shuffle: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                dataset, split,
                self._batch_size, self._seq_len, self._shuffle,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._2_nb_ch, self._seq_len, self._feat_len)
        if self._is_eval:
            label_shape = None
        else:
            label_shape = [
                (self._batch_size, self._seq_len, self._nb_classes),
                (self._batch_size, self._seq_len, self._nb_classes*2)
            ]
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        for filename in os.listdir(self._feat_dir):
            if int(filename[5]) in self._splits: # check which split the file belongs to
                self._filenames_list.append(filename)

        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
        self._nb_frames_file = temp_feat.shape[0]
        self._feat_len = temp_feat.shape[1] // self._2_nb_ch

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            self._label_len = temp_label.shape[-1]
            self._doa_len = (self._label_len - self._nb_classes)//self._nb_classes

        if self._per_file:
            self._batch_size = int(np.ceil(temp_feat.shape[0]/float(self._seq_len)))

        return

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """

        while 1:
            if self._shuffle:
                random.shuffle(self._filenames_list)

            # Ideally this should have been outside the while loop. But while generating the test data we want the data
            # to be the same exactly for all epoch's hence we keep it here.
            self._circ_buf_feat = deque()
            self._circ_buf_label = deque()

            file_cnt = 0
            if self._is_eval:
                for i in range(self._nb_total_batches):
                    # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                    # circular buffer. If not keep refilling it.
                    while len(self._circ_buf_feat) < self._batch_seq_len:
                        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                        for row_cnt, row in enumerate(temp_feat):
                            self._circ_buf_feat.append(row)

                        # If self._per_file is True, this returns the sequences belonging to a single audio recording
                        if self._per_file:
                            extra_frames = self._batch_seq_len - temp_feat.shape[0]
                            extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                            for row_cnt, row in enumerate(extra_feat):
                                self._circ_buf_feat.append(row)

                        file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                    feat = np.zeros((self._batch_seq_len, self._feat_len * self._2_nb_ch))
                    for j in range(self._batch_seq_len):
                        feat[j, :] = self._circ_buf_feat.popleft()
                    feat = np.reshape(feat, (self._batch_seq_len, self._feat_len, self._2_nb_ch))

                    # Split to sequences
                    feat = self._split_in_seqs(feat)
                    feat = np.transpose(feat, (0, 3, 1, 2))

                    yield feat

            else:
                for i in range(self._nb_total_batches):

                    # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                    # circular buffer. If not keep refilling it.
                    while len(self._circ_buf_feat) < self._batch_seq_len:
                        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                        temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                        for row_cnt, row in enumerate(temp_feat):
                            self._circ_buf_feat.append(row)
                            self._circ_buf_label.append(temp_label[row_cnt])

                        # If self._per_file is True, this returns the sequences belonging to a single audio recording
                        if self._per_file:
                            extra_frames = self._batch_seq_len - temp_feat.shape[0]
                            extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                            extra_labels = np.zeros((extra_frames, temp_label.shape[1]))
                            extra_labels[:, self._nb_classes:2 * self._nb_classes] = self._default_azi
                            extra_labels[:, 2 * self._nb_classes:] = self._default_ele

                            for row_cnt, row in enumerate(extra_feat):
                                self._circ_buf_feat.append(row)
                                self._circ_buf_label.append(extra_labels[row_cnt])

                        file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                    feat = np.zeros((self._batch_seq_len, self._feat_len * self._2_nb_ch))
                    label = np.zeros((self._batch_seq_len, self._label_len))
                    for j in range(self._batch_seq_len):
                        feat[j, :] = self._circ_buf_feat.popleft()
                        label[j, :] = self._circ_buf_label.popleft()
                    feat = np.reshape(feat, (self._batch_seq_len, self._feat_len, self._2_nb_ch))

                    # Split to sequences
                    feat = self._split_in_seqs(feat)
                    feat = np.transpose(feat, (0, 3, 1, 2))
                    label = self._split_in_seqs(label)

                    # Get azi/ele in radians
                    azi_rad = label[:, :, self._nb_classes:2 * self._nb_classes] * np.pi / 180
                    # ele_rad = label[:, :, 2 * self._nb_classes:] * np.pi / 180

                    # rescaling the elevation data from [-def_elevation def_elevation] to [-180 180] to keep them in the
                    # range of azimuth angle
                    ele_rad = label[:, :, 2 * self._nb_classes:] * np.pi / self._default_ele

                    label = [
                        label[:, :, :self._nb_classes],  # SED labels
                        np.concatenate((azi_rad, ele_rad), -1)  # DOA labels in radians
                         ]

                    yield feat, label

    def _split_in_seqs(self, data):
        if len(data.shape) == 1:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :, :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_default_elevation(self):
        return self._default_ele

    def get_azi_ele_list(self):
        return self._feat_cls.get_azi_ele_list()

    def get_list_index(self, azi, ele):
        return self._feat_cls.get_list_index(azi, ele)

    def get_matrix_index(self, ind):
        return self._feat_cls.get_matrix_index(ind)

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_classes(self):
        return self._feat_cls.get_classes()
    
    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls._max_frames

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()



