import argparse
import logging

import sys
import sklearn
from utils.utils import *
from utils.constants import *


def fit_classifier(dataset, classifier_name, output_directory):
    x_train = dataset[0]
    y_train = dataset[1]
    x_test = dataset[2]
    y_test = dataset[3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save original y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


def run(root_dir, classifiers, archive2dataset, iteration):
    for classifier_name in classifiers:
        for archive_name in archive2dataset:
            logging.info('Evaluating classifier {} on datasets {} from archive {}. Iteration #{}'
                         .format(classifier_name, ", ".join(archive2dataset[archive_name]), archive_name, iteration))

            datasets_dict = read_datasets(root_dir, archive_name, archive2dataset[archive_name])
            i = '_itr_{}'.format(iteration)

            tmp_output_directory = os.path.join(root_dir,'results', classifier_name, archive_name + i)
            for dataset_name in datasets_dict:
                print('\t\t\tdataset_name: ', dataset_name)
                output_directory = os.path.join(tmp_output_directory, dataset_name)
                create_directory(output_directory)

                fit_classifier(datasets_dict[dataset_name], classifier_name, output_directory)

                print('\t\t\t\tDONE')

                # the creation of this directory means
                create_directory(output_directory + '/DONE')


# ############################################## main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-op", "--operation", type=str, required=True,
                        help="Defines the operation that should be executed. Can be one of those:\n"
                             "\trun_all: Run all models on all datasets.\n"
                             "\trun_on_dataset: Run all models on a specific dataset.\n"
                             "\trun_model: Run one specific model on all datasets."
                             "\ttransform_mts_to_npy_format: Transform whole MTS dataset to numpy files.\n"
                             "\ttransform_ucr_to_npy_format: Transform whole UCR dataset to numpy files.\n"
                             "\tvisualize_filter: ...\n"
                             "\tviz_for_survey_paper: ...\n"
                             "\tviz_cam: ...\n"
                             "\tgenerate_results_csv: ...\n")
    parser.add_argument("-r", "--root-dir", type=str, required=True, help="Root data directory.")
    parser.add_argument("-a", "--archive", type=str, help="Archive name.")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name.")
    parser.add_argument("-m", "--model", type=str, help="Model name.")
    parser.add_argument("-i", "--iteration", type=int, help="Iteration number.")
    args = parser.parse_args()

    op = args.operation
    root_dir = args.root_dir
    iteration = args.iteration if args.iteration else 0

    if op == 'run_all':
        for classifier_name in CLASSIFIERS:
            run(root_dir, classifier_name, dataset_names_for_archive, iteration)
    elif op == "run_on_dataset":
        if args.archive and args.dataset:
            archive = args.archive
            dataset = args.dataset
            for classifier_name in CLASSIFIERS:
                run(root_dir, classifier_name, {archive: [dataset]}, iteration)
        else:
            logging.error("For operation mode {} parameters archive and dataset must be defined.".format(op))
    elif op == "run_model":
        if args.model:
            model = args.model
            run(root_dir, model, dataset_names_for_archive, iteration)
        else:
            logging.error("For operation mode {} parameter model must be defined.".format(op))
    elif op == 'transform_mts_to_npy_format':
        source_root_directory = root_dir + '/archives/mts_mat/'
        output_root_directory = root_dir + '/archives/' + MTS_ARCHIVE + '/'
        transform_mts_to_npy_format(source_root_directory, output_root_directory)
    elif op == 'transform_ucr_to_npy_format':
        transform_ts_to_npy_format(root_dir, UCR_UV_ARCHIVE, root_dir)
        transform_ts_to_npy_format(root_dir, UCR_MV_ARCHIVE, root_dir)
    elif op == 'visualize_filter':
        visualize_filter(root_dir)
    elif op == 'viz_for_survey_paper':
        viz_for_survey_paper(root_dir)
    elif op == 'viz_cam':
        viz_cam(root_dir)
    elif op == 'generate_results_csv':
        res = generate_results_csv('results.csv', root_dir)
        print(res.to_string())
    else:
        logging.error("Invalid operation parameter {}".format(op))


if __name__ == '__main__':
    main()



