# Extracts the features, labels, and normalizes the development and evaluation split features.
# NOTE: Change the dataset_dir and feat_label_dir path accordingly

import cls_feature_class

process_str = 'dev, eval'  # 'dev' or 'eval' will extract features for the respective set accordingly
#  'dev, eval' will extract features of both sets together

dataset_name = 'foa'  # 'foa' -ambisonic or 'mic' - microphone signals
dataset_dir = '/proj/asignal/DCASE2019/dataset/'   # Base folder containing the foa/mic and metadata folders
feat_label_dir = '/proj/asignal/DCASE2019/dataset/feat_label_tmp/'  # Directory to dump extracted features and labels


if 'dev' in process_str:
    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(dataset=dataset_name, dataset_dir=dataset_dir,
                                                  feat_label_dir=feat_label_dir)

    # Extract features and normalize them
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # # Extract labels in regression mode
    dev_feat_cls.extract_all_labels()


if 'eval' in process_str:
    # -----------------------------Extract ONLY features for evaluation set-----------------------------
    eval_feat_cls = cls_feature_class.FeatureClass(dataset=dataset_name, dataset_dir=dataset_dir,
                                                   feat_label_dir=feat_label_dir, is_eval=True)

    # Extract features and normalize them
    eval_feat_cls.extract_all_feature()
    eval_feat_cls.preprocess_features()

