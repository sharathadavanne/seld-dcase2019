# Script to test the SELD metrics, specifically,
# a) the DOA error when estimating the in spherical vs cartesian format
# b) Calculating the SED, DOA and SELD error with regression and classification based labels.

import os
import numpy as np
import sys
sys.path.append(os.path.join(sys.path[0], '..'))
import cls_feature_class
from metrics import evaluation_metrics


def description_file_to_regression_label_format(_feat_cls, _desc_file_dict):
    _labels = _feat_cls.get_labels_for_file(_desc_file_dict)
    return _labels


def description_file_to_classification_label_format(_feat_cls, _desc_file_dict):
    _labels = _feat_cls.get_clas_labels_for_file(_desc_file_dict)
    return _labels


# # ----------- TEST SPHERICAL VS CARTESIAN CENTRAL ANGLE ERROR ---------------
print('---- TEST SPHERICAL VS CARTESIAN CENTRAL ANGLE ERROR ----')

# Test spherical coordinates, located on unit sphere
az1, ele1, d1 = 1.3, -1.1, 1
az2, ele2, d2 = -0.2, 0.2, 1

# convert to cartesian and calculate distance in cartesian domain
x1, y1, z1 = evaluation_metrics.sph2cart(az1, ele1, d1)
x2, y2, z2 = evaluation_metrics.sph2cart(az2, ele2, d2)

# check if sph2cart and cart2sph is correct. this should print the values of az* and ele*
print(np.array(evaluation_metrics.cart2sph(x1, y1, z1))) # should be same as az1, ele1, d1
print(np.array(evaluation_metrics.cart2sph(x2, y2, z2))) # should be same as az2, ele2, d2

# angular distance in cartesian domain
cart_dist = evaluation_metrics.distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)

# angular distance in spherical domain
sph_dist = evaluation_metrics.distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

# the two distances should be identical
print('sph_dist: {}'.format(sph_dist))
print('cart_dist: {}'.format(cart_dist))
print('\n\n')



# # ----------- TEST SED, DOA and SELD METRICS ---------------
# In this section we read two random description files, and compute the metrics for two cases.
# Case 1 when the description files are identical which correlates to perfect results from an ideal SELD method.
# Case 2 when the description files are different, this correlates to a classifier with imperfections.
#
# The ideal way of calculating the SELD metrics is using the output format as seen in
# compute_seld_metrics_from_output_format_dict() of evaluation_metrics.py
# This is because when approaching DOA estimation as regression task similar to baseline, we estimate only one DOA for a
# given sound class. This is not ideal, because in the dataset, a particular sound class, can occur more than once at
# the same time in different spatial locations. But since the baseline method doesnt support it, the training and
# testing labels contain only one instance of these overlapping same class sound events. Thus the actual DOA score
# calculated using compute_doa_scores_regr() while training the baseline are sub-optimal. Hence we calculate the
# actual SELD metrics after the training is completed.
# If you are implementing DOA estimation using classification approach as shown in the examples below, you will
# be using compute_doa_scores_clas() for DOA metric calculation, which is the right way, so you will be calculating the
# correct SELD metrics during training as well.
#
# This problem in DOA metric occurs for regression scenario only when the number of overlapping events in the scene is
# more than one. The studied dataset has both scenarios of upto one and upto two overlapping sound events.

# see how the scores of regression/classification change with max one and max two overlapping sound events
ov = 1  # max overlap 1 or 2

if ov == 1:
    # When there is maximum one sound event at a given time, both regression and classification approach of DOA gives
    # identical DOA results.
    gt_desc_file = 'test_files/split1_ir0_ov1_1.csv'
    pred_desc_file = 'test_files/split1_ir1_ov1_21.csv'
else:
    # The regression and classification approach for DOA estimation gives different DOA results in sound scenes with
    # ov > 1. This is because in regression format of DOA estimation, a sound scene where a single class overlaps by
    # itself is not supported. Hence, in order to have similar results with regression and classification approach,
    # we use the DCASE 2019 output format.

    gt_desc_file = 'test_files/split1_ir0_ov2_11.csv'
    pred_desc_file = 'test_files/split1_ir1_ov2_31.csv'


# Load feature class. Also works with preloaded data_generator class. Like used in the seld.py code to compute metrics
feat_cls = cls_feature_class.FeatureClass()
hop_sec = feat_cls.get_hop_len_sec()
unique_classes = feat_cls.get_classes()
nb_classes = len(unique_classes)

gt_desc_file_dict = feat_cls.read_desc_file(gt_desc_file)
pred_desc_file_dict = feat_cls.read_desc_file(pred_desc_file)


# # ------------------------- DCASE 2019 OUTPUT FORMAT FROM DESCRIPTION FILE ------------------------
# # Example of obtaining the DCASE output format from the csv description file
output_file = '{}_out.csv'.format(gt_desc_file[:-4])
desc_file_dict = feat_cls.read_desc_file(gt_desc_file, in_sec=True)
output_format_dict = evaluation_metrics.description_file_to_output_format(desc_file_dict, unique_classes, hop_sec)


# --------------- REGRESSION FORMAT DOA LABELS SCORING -----------------------------------------
print('---- REGRESSION FORMAT DOA LABELS SCORING -----')
# Obtain regression labels from csv description format
gt_regr_labels = description_file_to_regression_label_format(feat_cls, gt_desc_file_dict)


print('\nPERFECT SELD RESULTS')
# use regression labels to score. When the predicted labels are prefect.
gt_sed_labels = gt_regr_labels[:, :nb_classes]
gt_doa_labels = gt_regr_labels[:, nb_classes:] * np.pi / 180.

er, f = evaluation_metrics.compute_sed_scores(gt_sed_labels, gt_sed_labels, feat_cls.nb_frames_1s())
doa_err, frame_recall, d1, d2, d3, d4 = evaluation_metrics.compute_doa_scores_regr(gt_doa_labels, gt_doa_labels, gt_sed_labels, gt_sed_labels)

print('SED metrics: er: {}, f:{}'.format(er, f))
print('DOA metrics: doa error: {}, frame recall:{}'.format(doa_err, frame_recall))

# Convert regression labels to DCASE 2019 output format
gt_regr_output_file = '{}_regr_out.csv'.format(gt_desc_file[:-4])
gt_regression_output_format_dict = evaluation_metrics.regression_label_format_to_output_format(feat_cls, gt_regr_labels[:, :nb_classes], gt_regr_labels[:, nb_classes:])
evaluation_metrics.write_output_format_file(gt_regr_output_file, gt_regression_output_format_dict)

print('\nPERFECT SELD RESULTS COMPUTED ON OUTPUT FORMAT DICTIONARY')
seld_scr, er, f, doa_err, frame_recall = evaluation_metrics.compute_seld_metrics_from_output_format_dict(gt_regression_output_format_dict, gt_regression_output_format_dict, feat_cls)
print('SELD score: {}'.format(seld_scr))
print('SED metrics: er: {}, f:{}'.format(er, f))
print('DOA metrics: doa error: {}, frame recall:{}'.format(doa_err, frame_recall))

print('\nIMPERFECT SELD RESULTS')
# use regression labels to score. When the predicted labels are imperfect.
pred_regr_labels = description_file_to_regression_label_format(feat_cls, pred_desc_file_dict)
pred_regr_output_file = '{}_regr_out.csv'.format(pred_desc_file[:-4])

pred_sed_labels = pred_regr_labels[:, :nb_classes]
pred_doa_labels = pred_regr_labels[:, nb_classes:] * np.pi / 180.

er, f = evaluation_metrics.compute_sed_scores(pred_sed_labels, gt_sed_labels, feat_cls.nb_frames_1s())
doa_err, frame_recall, d1, d2, d3, d4 = evaluation_metrics.compute_doa_scores_regr(pred_doa_labels, gt_doa_labels, pred_sed_labels, gt_sed_labels)

print('SED metrics: er: {}, f:{}'.format(er, f))
print('DOA metrics: doa error: {}, frame recall:{}'.format(doa_err, frame_recall))


print('\nIMPERFECT SELD RESULTS COMPUTED ON OUTPUT FORMAT DICTIONARY')
pred_regression_output_format_dict = evaluation_metrics.regression_label_format_to_output_format(feat_cls, pred_regr_labels[:, :nb_classes], pred_regr_labels[:, nb_classes:])
evaluation_metrics.write_output_format_file(pred_regr_output_file, pred_regression_output_format_dict)
seld_scr, er, f, doa_err, frame_recall = evaluation_metrics.compute_seld_metrics_from_output_format_dict(pred_regression_output_format_dict, gt_regression_output_format_dict, feat_cls)
print('SELD score: {}'.format(seld_scr))
print('SED metrics: er: {}, f:{}'.format(er, f))
print('DOA metrics: doa error: {}, frame recall:{}'.format(doa_err, frame_recall))


# # ------------------------ CLASSIFICATION FORMAT DOA LABELS SCORING --------------------------
print('\n\n---- CLASSIFICATION FORMAT DOA LABELS SCORING -----')
gt_clas_output_file = '{}_clas_out.csv'.format(gt_desc_file[:-4])
gt_clas_labels = description_file_to_classification_label_format(feat_cls, gt_desc_file_dict)

print('\nPERFECT SELD RESULTS')
# get only the SED labels
gt_sed_class_labels = gt_clas_labels.max(2)
er, f = evaluation_metrics.compute_sed_scores(gt_sed_class_labels, gt_sed_class_labels, feat_cls.nb_frames_1s())

doa_err, frame_recall, d1, d2, d3, d4 = evaluation_metrics.compute_doa_scores_clas(gt_clas_labels, gt_clas_labels, feat_cls)

print('SED metrics: er: {}, f:{}'.format(er, f))
print('DOA metrics: doa error: {}, frame recall:{}'.format(doa_err, frame_recall))


print('\nIMPERFECT SELD RESULTS')
# use classification labels to score. When the predicted labels are imperfect.
pred_clas_labels = description_file_to_classification_label_format(feat_cls, pred_desc_file_dict)
pred_clas_output_file = '{}_clas_out.csv'.format(pred_desc_file[:-4])
pred_clas_output_format_dict = evaluation_metrics.classification_label_format_to_output_format(feat_cls, pred_clas_labels)
evaluation_metrics.write_output_format_file(pred_clas_output_file, pred_clas_output_format_dict)

# get only the SED labels
pred_sed_clas_labels = pred_clas_labels.max(2)
er, f = evaluation_metrics.compute_sed_scores(pred_sed_clas_labels, gt_sed_class_labels, feat_cls.nb_frames_1s())
doa_err, frame_recall, d1, d2, d3, d4 = evaluation_metrics.compute_doa_scores_clas(pred_clas_labels, gt_clas_labels, feat_cls)
print('SED metrics: er: {}, f:{}'.format(er, f))
print('DOA metrics: doa error: {}, frame recall:{}'.format(doa_err, frame_recall))

print('\nIMPERFECT SELD RESULTS COMPUTED ON OUTPUT FORMAT DICTIONARY')
gt_clas_output_format_dict = evaluation_metrics.classification_label_format_to_output_format(feat_cls, gt_clas_labels)
evaluation_metrics.write_output_format_file(gt_clas_output_file, gt_clas_output_format_dict)
seld_scr, er, f, doa_err, frame_recall = evaluation_metrics.compute_seld_metrics_from_output_format_dict(pred_clas_output_format_dict, gt_clas_output_format_dict, feat_cls)
print('SELD score: {}'.format(seld_scr))
print('SED metrics: er: {}, f:{}'.format(er, f))
print('DOA metrics: doa error: {}, frame recall:{}'.format(doa_err, frame_recall))
