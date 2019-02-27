import os
import evaluation_metrics
import cls_feature_class
import numpy as np
from IPython import embed


def get_nb_files(_pred_file_list, _group='split'):
    _group_ind = {'split': 5, 'ir': 9, 'ov': 13}
    _cnt_dict = {}
    for _filename in _pred_file_list:

        if _group == 'all':
            _ind = 0
        else:
            _ind = int(_filename[_group_ind[_group]])

        if _ind not in _cnt_dict:
            _cnt_dict[_ind] = []
        _cnt_dict[_ind].append(_filename)

    return _cnt_dict


# --------------------------- MAIN SCRIPT STARTS HERE -------------------------------------------

# Metric evaluation at a fixed hop length of 20 ms (=0.02 seconds) and 3000 frames for a 60s audio

# INPUT DIRECTORY
ref_desc_files = '/home/adavanne/taitoSharedData/DCASE2019/dataset/metadata_dev' # reference description directory location
pred_output_format_files = '/home/adavanne/taitoWorkDir/SELD_DCASE2019/results/64_mic_dev' # predicted output format directory location

# Load feature class
feat_cls = cls_feature_class.FeatureClass()
max_frames = feat_cls.get_nb_frames()
unique_classes = feat_cls.get_classes()
nb_classes = len(unique_classes)
azi_list, ele_list = feat_cls.get_azi_ele_list()

# collect reference files info
ref_files = os.listdir(ref_desc_files)
nb_ref_files = len(ref_files)

# collect predicted files info
pred_files = os.listdir(pred_output_format_files)
nb_pred_files = len(pred_files)

if nb_ref_files != nb_pred_files:
    print('ERROR: Mismatch. Reference has {} and prediction has {} files'.format(nb_ref_files, nb_pred_files))
    exit()

# Load evaluation metric class
eval = evaluation_metrics.SELDMetrics(nb_frames_1s=feat_cls.nb_frames_1s(), data_gen=feat_cls)

# Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)
score_type_list = [ 'all', 'split', 'ov', 'ir']

print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

for score_type in score_type_list:
    print('\n\n---------------------------------------------------------------------------------------------------')
    print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type=='all' else 'score per {}'.format(score_type)))
    print('---------------------------------------------------------------------------------------------------')

    split_cnt_dict = get_nb_files(pred_files, _group=score_type) # collect files corresponding to score_type

    # Calculate scores across files for a given score_type
    for split_key in np.sort(list(split_cnt_dict)):
        eval.reset()    # Reset the evaluation metric parameters
        for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
            # Load predicted output format file
            pred_dict = evaluation_metrics.load_output_format_file(os.path.join(pred_output_format_files, pred_file))

            # Load reference description file
            gt_desc_file_dict = feat_cls.read_desc_file(os.path.join(ref_desc_files, pred_file.replace('.npy', '.csv')))

            # Generate classification labels for SELD
            gt_labels = feat_cls.get_clas_labels_for_file(gt_desc_file_dict)
            pred_labels = evaluation_metrics.output_format_dict_to_classification_labels(pred_dict, feat_cls)

            # Calculated SED and DOA scores
            eval.update_sed_scores(pred_labels.max(2), gt_labels.max(2))
            eval.update_doa_scores(pred_labels, gt_labels)

        # Overall SED and DOA scores
        er, f = eval.compute_sed_scores()
        doa_err, frame_recall = eval.compute_doa_scores()
        seld_scr = evaluation_metrics.compute_seld_metric([er, f], [doa_err, frame_recall])

        print('\nAverage score for {} {} data'.format(score_type, 'fold' if score_type=='all' else split_key))
        print('SELD score: {}'.format(seld_scr))
        print('SED metrics: er: {}, f:{}'.format(er, f))
        print('DOA metrics: doa error: {}, frame recall:{}'.format(doa_err, frame_recall))

