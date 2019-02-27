# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        dataset_dir='/proj/asignal/DCASE2019/dataset/',  # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATH
        feat_label_dir='/proj/asignal/DCASE2019/dataset/feat_label/',  # Directory to dump extracted features and labels
        model_dir='models/',   # Dumps the trained models and training curves in this folder
        dcase_output=True,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir='results/',  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        # DNN MODEL PARAMETERS
        sequence_length=128,        # Feature sequence length
        batch_size=16,              # Batch size
        dropout_rate=0,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        pool_size=[8, 8, 4],        # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
        rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 50.],     # [sed, doa] weight for scaling the DNN outputs
        nb_epochs=50,               # Train for maximum epochs
        epochs_per_fit=5,           # Number of epochs per fit

    )
    params['patience'] = int(0.1 * params['nb_epochs'])     # Stop training if patience is reached

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '50':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'

    elif argv == '51':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'

    elif argv == '52':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'

    elif argv == '53':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'

    #test scaled elevation values
    elif argv == '54':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'

    elif argv == '55':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'

    elif argv == '56':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'

    elif argv == '57':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'


    #test scaled elevation values with 2048 fft
    elif argv == '64':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'

    elif argv == '65':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'

    elif argv == '66':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'

    elif argv == '67':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'


    elif argv == '60':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['dropout_rate'] = 0.25
    elif argv == '61':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['dropout_rate'] = 0.25
    elif argv == '62':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['dropout_rate'] = 0.25
    elif argv == '63':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['dropout_rate'] = 0.25



    elif argv == '2':
        params['loss_weights'] = [1., 1.]

    elif argv == '3':
        params['loss_weights'] = [1., 5.]

    elif argv == '4':
        params['loss_weights'] = [1., 50.]

    elif argv == '5':
        params['loss_weights'] = [1., 500.]

    elif argv == '6':
        params['nb_cnn2d_filt'] = 32

    elif argv == '7':
        params['nb_cnn2d_filt'] = 64

    elif argv == '8':
        params['nb_cnn2d_filt'] = 128

    elif argv == '9':
        params['nb_cnn2d_filt'] = 256

    elif argv == '10':
        params['dropout_rate'] = 0

    elif argv == '11':
        params['dropout_rate'] = 0.25

    elif argv == '12':
        params['dropout_rate'] = 0.5

    elif argv == '13':
        params['dropout_rate'] = 0.75

    elif argv == '14':
        params['rnn_size'] = [128]

    elif argv == '15':
        params['rnn_size'] = [64, 64]

    elif argv == '16':
        params['rnn_size'] = [64]

    elif argv == '17':
        params['rnn_size'] = [32, 32]

    elif argv == '18':
        params['rnn_size'] = [32]

    elif argv == '19':
        params['rnn_size'] = [16, 16]

    elif argv == '20':
        params['rnn_size'] = [16]

    elif argv == '21':
        params['sequence_length'] = 32

    elif argv == '22':
        params['sequence_length'] = 64

    elif argv == '23':
        params['sequence_length'] = 128

    elif argv == '24':
        params['sequence_length'] = 256

    elif argv == '25':
        params['batch_size'] = 8

    elif argv == '26':
        params['batch_size'] = 16

    elif argv == '27':
        params['batch_size'] = 32

    elif argv == '28':
        params['batch_size'] = 64

    elif argv == '29':
        params['nb_epochs'] = 2

    elif argv == '30':
        params['pool_size'] = [8, 8, 2]
        params['rnn_size'] = []

    elif argv == '31':
        params['pool_size'] = [8, 4, 2, 2]
        params['rnn_size'] = []

    elif argv == '32':
        params['pool_size'] = [4, 4, 2, 2, 2]
        params['rnn_size'] = []

    elif argv == '33':
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []

    elif argv == '34':
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['dropout_rate'] = 0.25

    elif argv == '35':
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['dropout_rate'] = 0.5

    elif argv == '36':
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['loss_weights'] = [1., 1.]
    elif argv == '37':
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['loss_weights'] = [1., 5.]

    elif argv == '38':
        params['nb_cnn2d_filt'] = 32
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['dropout_rate'] = 0.25

    elif argv == '39':
        params['nb_cnn2d_filt'] = 32
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['dropout_rate'] = 0.5

    elif argv == '40':
        params['nb_cnn2d_filt'] = 32
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['loss_weights'] = [1., 1.]
    elif argv == '41':
        params['nb_cnn2d_filt'] = 32
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
        params['loss_weights'] = [1., 5.]

    elif argv == '42':
        params['nb_cnn2d_filt'] = 64
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
    elif argv == '43':
        params['nb_cnn2d_filt'] = 32
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
    elif argv == '44':
        params['nb_cnn2d_filt'] = 16
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1
    elif argv == '45':
        params['nb_cnn2d_filt'] = 8
        params['pool_size'] = [4, 2, 2, 2, 2, 2]
        params['rnn_size'] = []
        params['epochs_per_fit'] = 1

    # Quick test
    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['epochs_per_fit'] = 1

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
