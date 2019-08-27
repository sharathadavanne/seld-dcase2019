
# DCASE 2019: Sound event localization and detection (SELD) task
> The DCASE 2019 Challenge has now ended, the results of all the submissions can be seen [here](http://dcase.community/challenge2019/task-sound-event-localization-and-detection-results). 

> Check [our new repository](https://github.com/sharathadavanne/seld-net) for sound event localization, detection and tracking of multiple stationary and moving sources. This repository also includes datasets with stationary sources in multi-reverberant scenario synthesized using Ambisonic and Circular Array formats. Additionally, it includes datasets with sources moving in varying angular velocities in Ambisonic format.

Sound event localization and detection (SELD) is the combined task of identifying the temporal onset and offset of a sound event, tracking the spatial location when active, and further associating a textual label describing the sound event. As part of [DCASE 2019](http://dcase.community/challenge2019/index), we are organizing an [SELD task](http://dcase.community/challenge2019/task-sound-event-localization-and-detection) with a [multi-room reverberant dataset synthesized using real-life impulse response (IR) collected at five different environments](https://arxiv.org/pdf/1905.08546.pdf 'Paper on Arxiv'). This github page shares the benchmark method, SELDnet, and the dataset for the task. The paper describing the SELDnet can be found on [IEEExplore](https://ieeexplore.ieee.org/document/8567942 'Paper on IEEE Xplore') and on [Arxiv](https://arxiv.org/pdf/1807.00129.pdf 'Paper on Arxiv'). The dataset, baseline method and benchmark scores have been described in the task paper available [here](https://arxiv.org/pdf/1905.08546.pdf 'Paper on Arxiv').
   
If you are using this code or the datasets in any format, then please consider citing the following two papers

> Sharath Adavanne, Archontis Politis and Tuomas Virtanen, "A multi-room reverberant dataset for sound event localization and detection" submitted in the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2019)

> Sharath Adavanne, Archontis Politis, Joonas Nikunen and Tuomas Virtanen, "Sound event localization and detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected Topics in Signal Processing (JSTSP 2018)

## More about SELDnet
The SELDnet architecture is as shown below. The input is the multichannel audio, from which the phase and magnitude components are extracted and used as separate features. The proposed method takes a sequence of consecutive spectrogram frames as input and predicts all the sound event classes active for each of the input frame along with their respective spatial location, producing the temporal activity and DOA trajectory for each sound event class. In particular, a convolutional recurrent neural network (CRNN) is used to map the frame sequence to the two outputs in parallel. At the first output, SED is performed as a multi-label multi-class classification task, allowing the network to simultaneously estimate the presence of multiple sound events for each frame. At the second output, DOA estimates in the continuous 3D space are obtained as a multi-output regression task, where each sound event class is associated with two regressors that estimate the spherical coordinates azimuth (azi) and elevation (ele) of the DOA on a unit sphere around the microphone.

In the benchmark method, the variables in the image below have the following values, T = 128, M = 2048, C = 4, P = 64, MP<sub>1</sub> = MP<sub>2</sub> = 8, MP<sub>3</sub> = 4, Q = R = 128, N = 11.

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2019/blob/master/images/DCASE2019_SELDnet.png" width="600" title="SELDnet Architecture">
</p>



The SED output of the network is in the continuous range of [0 1] for each sound event in the dataset, and this value is thresholded to obtain a binary decision for the respective sound event activity as shown in figure below. Finally, the respective DOA estimates for these active sound event classes provide their spatial locations.

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2019/blob/master/images/DCASE2019_SELDnet_output.png" width="400" title="SELDnet output format">
</p>

The figure below visualizes the SELDnet input and outputs for one of the recordings in the dataset. The horizontal-axis of all sub-plots for a given dataset represents the same time frames, the vertical-axis for spectrogram sub-plot represents the frequency bins, vertical-axis for SED reference and prediction sub-plots represents the unique sound event class identifier, and for the DOA reference and prediction sub-plots, it represents the azimuth and elevation angles in degrees. The figures represents each sound event class and its associated DOA outputs with a unique color. Similar plot can be visualized on your results using the [provided script](misc_files/visualize_SELD_output.py).

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2019/blob/master/images/SELDnet_output.png" width="1200" title="SELDnet input and output visualization">
</p>

## DATASETS

The participants can choose either of the two or both the following datasets,

 * **TAU Spatial Sound Events 2019 - Ambisonic**
 * **TAU Spatial Sound Events 2019 - Microphone Array**

These datasets contain recordings from an identical scene, with **TAU Spatial Sound Events 2019 - Ambisonic** providing four-channel First-Order Ambisonic (FOA) recordings while  **TAU Spatial Sound Events 2019 - Microphone Array** provides four-channel directional microphone recordings from a tetrahedral array configuration. Both formats are extracted from the same microphone array, and additional information on the spatial characteristics of each format can be found below. The participants can choose one of the two, or both the datasets based on the audio format they prefer. Both the datasets, consists of a development and evaluation set. The development set consists of 400, one minute long recordings sampled at 48000 Hz, divided into four cross-validation splits of 100 recordings each. The evaluation set consists of 100, one-minute recordings. These recordings were synthesized using spatial room impulse response (IRs) collected from five indoor locations, at 504 unique combinations of azimuth-elevation-distance. Furthermore, in order to synthesize the recordings the collected IRs were convolved with [isolated sound events dataset from DCASE 2016 task 2](http://www.cs.tut.fi/sgn/arg/dcase2016/task-sound-event-detection-in-synthetic-audio#audio-dataset). Finally, to create a realistic sound scene recording, natural ambient noise collected in the IR recording locations was added to the synthesized recordings such that the average SNR of the sound events was 30 dB.

The eleven sound event classes used in the dataset and their corresponding index values required for the submission format are as following

| Sound class| Index |
| ----| ---- |
| knock | 0 |
| drawer | 1 |
| clearthroat | 2 |
| phone | 3 |
| keysDrop | 4 |
| speech | 5 |
| keyboard | 6 |
| pageturn | 7 |
| cough | 8 |
| doorslam | 9 |
| laughter | 10 |

More details on the recording procedure and dataset can be read on the [DCASE 2019 task webpage](http://dcase.community/challenge2019/task-sound-event-localization-and-detection) or on the [task description paper](https://arxiv.org/pdf/1905.08546.pdf 'Paper on Arxiv').

The two development datasets can be downloaded from the link - [**TAU Spatial Sound Events 2019 - Ambisonic and Microphone Array**, Development dataset (Version 2)](https://doi.org/10.5281/zenodo.2599196) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2599196.svg)](https://doi.org/10.5281/zenodo.2599196) 

> Dataset was updated on <strong>20 March 2019</strong> to remove labels of sound events that were missing in the audio (version 2). In order to update already downloaded dataset version 1, download only the <code>metadata_dev.zip</code> file from version 2.

The evaluation datasets can be downloaded from the link - [**TAU Spatial Sound Events 2019 - Ambisonic and Microphone Array**, Evaluation dataset](https://doi.org/10.5281/zenodo.3377088) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3377088.svg)](https://doi.org/10.5281/zenodo.3377088) 

> Dataset was updated on <strong>26 August 2019</strong>: Now that the task has ended, we are releasing the reference labels for the evaluation dataset (version 2).


## Getting Started

This repository consists of multiple Python scripts forming one big architecture used to train the SELDnet.
* The `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `parameter.py` script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The `cls_feature_class.py` script has routines for labels creation, features extraction and normalization.
* The `cls_data_generator.py` script provides feature + label data in generator mode for training.
* The `keras_model.py` script implements the SELDnet architecture.
* The `evaluation_metrics.py` script, implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and the DOA metrics explained in the paper.
* The `seld.py` is a wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.

Additionally, we also provide supporting scripts that help analyse the dataset and results.
 * `check_dataset_distribution.py` visualizes the dataset distribution in different configurations.
 * `visualize_SELD_output.py` script to visualize the SELDnet output
 * `test_SELD_metrics.py` test script to evaluate the different metrics employed


### Prerequisites

The provided codebase has been tested on python 2.7.10/3.5.3. and Keras 2.2.2./2.2.4


### Training the SELDnet

In order to quickly train SELDnet follow the steps below.

* For the chosen dataset (Ambisonic or Microphone), download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are Ambisonic dataset, then the 'base_folder/' should have two folders - 'foa_dev/' and 'metadata_dev/' after unzipping.

* Now update the respective dataset path in `parameter.py` script. For the above example, you will change `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped. Make sure this folder has sufficient space. For example if you use the baseline configuration, you will need about 160 GB in total just for the features and labels.

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. First, update the parameters in the script, check the python file for more comments. You can now run the script as shown below. This will dump the normalized features and labels here. Since feature extraction is a one-time thing, this script is standalone and does not use the `parameter.py` file.

```
python batch_feature_extraction.py
```

You can now train the SELDnet using default parameters using
```
python seld.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `parameter.py` script and call them as following
```
python seld.py <task-id> <job-id>
```
Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.

In order to get baseline results on the development set for Microphone array recordings, you can run the following command
```
python seld.py 2
```
Similarly, for Ambisonic format baseline results, run the following command
```
python seld.py 4
```

* By default, the code runs in `quick_test = True` mode. This trains the network for 2 epochs on only 2 mini-batches. Once you get to run the code sucessfully, set `quick_test = False` in `parameter.py` script and train on the entire data.

* The code also plots training curves, intermediate results and saves models in the `model_dir` path provided by the user in `parameter.py` file.

* In order to visualize the output of SELDnet and for submission of results, set `dcase_output=True` and provide `dcase_dir` directory. This will dump file-wise results in the directory, which can be individually visualized using `misc_files/visualize_SELD_output.py` script.

* Finally, the average development dataset score across the four folds can be obtained using `calculate_SELD_metrics.py` script. Provide the directory where you dumped the file-wise results above and the reference metadata folder. Check the comments in the script for more description.

## Results on development dataset


| Dataset | Error rate | F score| DOA error | Frame recall |
| ----| --- | --- | --- | --- |
| Ambisonic | 0.34 | 79.9 % | 28.5&deg; | 85.4 % |
| Microphone Array |0.35 | 80.0 % | 30.8&deg; | 84.0 % |

**Note:** The reported baseline system performance is not exactly reproducible due to varying setups. However, you should be able to obtain very similar results.

## DOA estimation: regression vs classification

The DOA estimation can be approached as both a regression or a classification task. In the baseline, it is handled as regression task. In case you plan to use a classification approach check the `test_SELD_metrics.py` script in misc_files folder. It implements a classification version of DOA and also uses a corresponding metric function.


## Submission

* Before submission, make sure your SELD results are correct by visualizing the results using `misc_files/visualize_SELD_output.py` script
* Make sure the file-wise output you are submitting is produced at 20 ms hop length. At this hop length a 60 s audio file has 3000 frames.
* Calculate your development score for the four splits using the `calculate_SELD_metrics.py` script. Check if the average results you are obtaining here is comparable to the results you were obtaining during training.

For more information on the submission file formats [check the website](http://dcase.community/challenge2019/task-sound-event-localization-and-detection#submission)

## License

Except for the contents in the `metrics` folder that have [MIT License](metrics/LICENSE.md). The rest of the repository is licensed under the [TAU License](LICENSE.md).

## Acknowledgments

The research leading to these results has received funding from the European Research Council under the European Unions H2020 Framework Programme through ERC Grant Agreement 637422 EVERYSOUND.
