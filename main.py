import os
import pandas as pd
from pandas import DataFrame
import opensmile
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
import numpy as np
import random
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report, \
    mean_absolute_error, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Tuple
from pathlib import Path
import audiofile
from tqdm import tqdm
import graphviz
import shap
import matplotlib.pyplot as plt
import torchaudio
import warnings
import pyarrow as pa
import pickle
import copy


def cli_parser() -> Namespace:
    parser = ArgumentParser(description='Sami corpus')
    parser.add_argument('--datadir',
                        type=str,
                        default='./data')
    parser.add_argument('--datafile',
                        type=str,
                        default='data.csv')
    parser.add_argument('--train_set',
                        type=str,
                        choices=['train_small',
                                 'train_medium',
                                 'train_large'],
                        default='train_small')
    parser.add_argument('--split',
                        type=str,
                        default='split_small')
    parser.add_argument('--target_label',
                        type=str,
                        default='status')
    parser.add_argument('--feature_set',
                        type=str,
                        choices=['opensmile',
                                 'kaldi'],
                        default='kaldi')
    parser.add_argument('--feature_subset',
                        type=str,
                        choices=['mfccs',
                                 'fbanks',
                                 'None',
                                 'spectrograms'],
                        default='mfccs')
    parser.add_argument('--pooling_method',
                        type=str,
                        choices=['mean',
                                 'std',
                                 'None',
                                 'meanstd'],
                        default='meanstd')
    parser.add_argument('--classifier',
                        type=str,
                        choices=['lreg',
                                 'dtree',
                                 'rforest',
                                 'None'],
                        default='dtree')
    parser.add_argument('--feature_file',
                        type=str,
                        default='None')
    parser.add_argument('--portion',
                        type=float,
                        default='1.0')
    parser.add_argument('--train',
                        action=BooleanOptionalAction)
    parser.add_argument('--test',
                        action=BooleanOptionalAction)
    parser.add_argument('--predict',
                        action=BooleanOptionalAction)
    parser.add_argument('--preprocess_all',
                        action=BooleanOptionalAction)
    parser.add_argument('--explain',
                        action=BooleanOptionalAction)
    parser.add_argument('-cp',
                        '--checkpoint',
                        type=str,
                        nargs='+',
                        dest='checkpoint',
                        default='[]')
    parser.add_argument('-cpdir',
                        '--checkpoint_dir',
                        type=str,
                        dest='checkpoint_dir',
                        default='./checkpoints')
    parser.add_argument('--seed',
                        type=int,
                        default=3742783)
    return parser.parse_args()


def main() -> None:
    # parse arguments
    config = cli_parser()

    # set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # load datafile and drop NaNs
    df = pd.read_csv(config.datafile, sep='\t')

    # cleanup: force remove entries with a NaN present in target_label field
    df = df.dropna(how='any', subset=[config.target_label]).reset_index(drop=True)

    # make a first run over the entire dataset and extract features
    if config.preprocess_all:
        entire_dataset_features = extract_features(
            {'paths': [path if Path(path).is_absolute() else Path(config.datadir).joinpath(path) for path in
                       df['path'].to_list()], 'labels': df[config.target_label].to_list()},
            feature_set=config.feature_set,
            feature_subset=config.feature_subset,
            pooling_method=config.pooling_method
        )
        df['features'] = list(entire_dataset_features['features'])
        # TODO: add check for label consistency between files-features-labels

        # save file
        feature_file = f'features_{Path(config.datafile).stem}_{config.feature_set}_' \
                       f'{config.feature_subset}_{config.pooling_method}.full.pickle'
        print(f'\nSaving features in file {feature_file}...\n')
        with open(feature_file, 'wb') as handle:
            pickle.dump({'df': df, 'labels': entire_dataset_features['labels'],
                         'feature_names': entire_dataset_features['feature_names']}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\nFeatures saved!\n')

    # run only on a subset of the data
    if config.portion < 1.0:  # keep config.portion of the data
        drop_indices = np.random.choice(df.index, int(len(df)*(1.0-config.portion)), replace=False)
        df = df.drop(drop_indices).reset_index(drop=True)

    # set variables
    target_label = config.target_label
    name2label = {Path(path).name: label for (path, label) in zip(df['path'].to_list(), df[target_label].to_list())}
    label2name = {label: Path(path).name for (path, label) in zip(df['path'].to_list(), df[target_label].to_list())}

    if config.feature_file == 'None':
        # if not given a split create a random split
        print(f'\nGenerating data splits...')
        # TODO: fix option to add pre-defined split or remove entirely
        if len(config.split) > 0:
            train_data, test_data = random_split(df, splitx=[0.9, 0.1], label_dict=name2label, data_dir=config.datadir,
                                                 include_labels=['gender'])
            # train_data, test_data = balanced_split(df, n_speakers_out=4, named_speakers_out=None, label_dict=name2label,
            #                                        data_dir=config.datadir, equal_sets=True, include_labels=['sex'])
        else:
            # train_data, test_data = random_split(df, splitx=[0.9, 0.1], label_dict=name2label, data_dir=config.datadir,
            #                                      include_labels=['sex'])
            train_data, test_data = balanced_split(df, n_speakers_out=4, named_speakers_out=None, label_dict=name2label,
                                                   data_dir=config.datadir, equal_sets=True, include_labels=['sex'])
        # print stats
        print_dataset_stats(train_data, labels_to_print=['labels', 'gender'], desc='train')
        print_dataset_stats(test_data, labels_to_print=['labels', 'gender'], desc='test')

        print(f'\nExtracting features...')
        # extract features
        test_data_processed = extract_features(test_data,
                                               feature_set=config.feature_set,
                                               feature_subset=config.feature_subset,
                                               pooling_method=config.pooling_method)
        train_data_processed = extract_features(train_data,
                                                feature_set=config.feature_set,
                                                feature_subset=config.feature_subset,
                                                pooling_method=config.pooling_method)
        feature_names = train_data_processed['feature_names']
        print(f'Features extracted!\n')

        # remove NaNs
        X = []
        Y = []
        for featureset in [train_data_processed, test_data_processed]:
            idx = np.where(np.sum(np.isnan(featureset['features']), axis=1) > 0)[0]
            idx = np.array(list(set(np.arange(0, len(featureset['features']))) - set(idx)))
            X.append(featureset['features'][idx, :])
            Y.append(np.array(featureset['labels'])[idx].tolist())

        # save features (save in pickle or preferably in a non-serializable storage format such as arrow/parquet)
        feature_file = f'features_{Path(config.datafile).stem}_{config.feature_set}_' \
                       f'{config.feature_subset}_{config.pooling_method}_dataslice={config.portion}.pickle'
        print(f'\nSaving features in file {feature_file}...\n')
        with open(feature_file, 'wb') as handle:
            pickle.dump({'features': X, 'labels': Y, 'feature_names': train_data_processed['feature_names']}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\nFeatures saved!\n')
    else:
        try:
            with open(config.feature_file, "rb") as input_file:
                print(f'\nLoading file {config.feature_file}...')
                loaded_features = pickle.load(input_file)
                feature_names = loaded_features['feature_names']
                if 'features' in loaded_features.keys():
                    X = loaded_features['features']
                    Y = loaded_features['labels']
                else:
                    edf = loaded_features['df']
                    # edf = edf.drop(edf[edf['sex'] == 'f'].index).reset_index(drop=True)  # TODO: REMOVE THIS LINE
                    # if not given a split create a random split
                    # TODO: fix option to add pre-defined split or remove entirely
                    print(f'\nGenerating data splits...')
                    if len(config.split) > 0:
                        train_data, test_data = random_split(edf, splitx=[0.9, 0.1], label_dict=name2label,
                                                            data_dir=config.datadir)
                        # train_data, test_data = balanced_split(edf, n_speakers_out=6, named_speakers_out=None,
                        #                                        label_dict=name2label,
                        #                                        data_dir=config.datadir, equal_sets=True)
                    else:
                        # train_data, test_data = random_split(edf, splitx=[0.9, 0.1], label_dict=name2label,
                        #                                      data_dir=config.datadir)
                        train_data, test_data = balanced_split(edf, n_speakers_out=6, named_speakers_out=None,
                                                               label_dict=name2label,
                                                               data_dir=config.datadir, equal_sets=True)
                    X = [train_data['features'], test_data['features']]
                    Y = [train_data['labels'], test_data['labels']]
                    # print stats
                    print_dataset_stats(train_data, desc='train')
                    print_dataset_stats(test_data, desc='test')
                print(f'File loaded!')
        except FileNotFoundError as e:
            print('File does not exist:', e)
        except Exception as e:
            print('An error occurred:', e)

    # select classifier
    if config.classifier == 'dtree':
        clf = tree.DecisionTreeClassifier(random_state=config.seed)
    elif config.classifier == 'rforest':
        clf = RandomForestClassifier(n_estimators=100, random_state=config.seed)
    elif config.classifier == 'lreg':
        clf = LogisticRegression(random_state=config.seed)
    else:
        raise NotImplementedError('Classification method not implemented!')

    # train classifier
    print(f'\nTraining classifier...')
    clf = clf.fit(X[0], Y[0])
    print(f'Training completed!\n')
    y_pred = clf.predict(X[1])
    prediction_probs = clf.predict_proba(X[1])

    # training/test metrics
    labels = sorted(set(df[target_label].to_list()))
    label2ind = {label: idx for (idx, label) in enumerate(labels)}
    ind2label = {idx: label for (idx, label) in enumerate(labels)}
    mae_train = mean_absolute_error([label2ind[y] for y in Y[0]], [label2ind[y] for y in clf.predict(X[0])]).round(2)
    mae_test = mean_absolute_error([label2ind[y] for y in Y[1]], [label2ind[y] for y in y_pred]).round(2)
    print(f"Mean absolute error on training set: {mae_train}")
    print(f"Mean absolute error on test set: {mae_test}")

    # print metrics
    metrics = precision_recall_fscore_support(Y[1], y_pred)
    print(classification_report(Y[1], y_pred))
    print(confusion_matrix(Y[1], y_pred, labels=labels))

    # visualize feature importance
    if config.explain:
        # extract gini importance values from the tree
        importances_sk = clf.feature_importances_

        # decision tree-based criterion
        print('\nExtracting graph from decision tree...\n')
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names,
                                        class_names=labels, filled=True,
                                        rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        # graph.view()
        # shap values
        print('\nComputing shap values for decision tree...\n')
        # shap.initjs()
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X[0])
        shap_interaction_values = explainer.shap_interaction_values(X[0])
        print('\nGenerating figures...\n')
        shap.summary_plot(shap_values[1], X[0], feature_names=feature_names, plot_type='dot')
        f = plt.figure()
        shap.summary_plot(shap_values[1], X[0], feature_names=feature_names, max_display=10, plot_type='bar')
        # f.savefig("/summary_plot1.png", bbox_inches='tight', dpi=600)


def print_dataset_stats(dataset: Dict = None, labels_to_print: List[str] = ['labels'], desc: str = None) -> None:
    # defs
    counts = lambda x, name, target: np.sum([label == target for label in x[name]])
    percent = lambda x, y: (x / y) * 100

    # init vars
    n = len(dataset[labels_to_print[0]])

    # print
    print(f"\nStats for {desc} dataset:")
    for label in labels_to_print:
        discrete_labels = sorted(set(dataset[label]))
        print(
            f"\t{label.upper()}:\t" +
            f"\t-\t".join(
                [f"{sublabel}: {counts(dataset, label, sublabel)} ({percent(counts(dataset, label, sublabel), n):.2f}%)"
                 for sublabel in discrete_labels])
        )


def conditional_update_dict(dic: Dict = None, pdf: DataFrame = None, label: str = None, indices: List[int] = None) \
        -> None:
    return dic.update(**{label: [pdf.iloc[ind][label] for ind in indices]} if label in pdf.columns else {})


def update_dict(dic: Dict = None, pdf: DataFrame = None, label: str = None, indices: List[int] = None) -> None:
    return dic.update(**{label: [pdf.iloc[ind][label] for ind in indices]})


def random_split(data: DataFrame = None, splitx: List[float] = [0.9, 0.1], label_dict: Dict = None,
                 data_dir: str = None, include_labels: List[str] = []) -> Tuple[Dict, Dict]:
    """
    Returns a random split of data given a dataframe with the data
    :param include_labels: user specified labels to add in the returned data objects
    :param data_dir: path to speech data
    :param label_dict: dictionary with correspondence between data and labels
    :param data: DataFrame with data
    :param splitx: train / test split
    :return:
    """
    idx = np.random.permutation(len(data))
    train_idx = idx[0: int(len(idx) * splitx[0])]
    test_idx = idx[int(len(idx) * splitx[0]) + 1:]
    train_data = {'paths': [path if Path(path).is_absolute() else Path(data_dir).joinpath(path)
                            for path in data['path'].to_numpy()[train_idx]],
                  'labels': [label_dict[fname] for fname in
                             [Path(path).name for path in list(data['path'].to_numpy()[train_idx])]]}
    test_data = {'paths': [path if Path(path).is_absolute() else Path(data_dir).joinpath(path)
                           for path in data['path'].to_numpy()[test_idx]],
                 'labels': [label_dict[fname] for fname in
                            [Path(path).name for path in list(data['path'].to_numpy()[test_idx])]]}

    # add features if available
    conditional_update_dict(train_data, data, 'features', train_idx)
    conditional_update_dict(test_data, data, 'features', test_idx)

    # add vars if requested
    for label in include_labels:
        update_dict(train_data, data, label, train_idx)
        update_dict(test_data, data, label, test_idx)

    return train_data, test_data


def balanced_split(data: DataFrame = None, n_speakers_out: int = 4, named_speakers_out: Dict[List, List] = None,
                   label_dict: Dict = None, data_dir: str = None, equal_sets: bool = False,
                   include_labels: List[str] = []) -> Tuple[Dict, Dict]:
    """
    We want to split this based on speaker-dependent and speaker-independent criteria and save the split in a file.
    :param include_labels: user specified labels to add in the returned data objects
    :param equal_sets: force the utterance counts of each speaker to be equal in opposition and coalition
    :param named_speakers_out: give dictionary of named speakers to add in test.
        The expected format is the following: {'opposition': [], 'coalition':[]}.
        Note: since opposition and coalition speakers are present in both, the speaker data in the coalition list
        will be preserved and removed for the opposition and vice versa.
        Note: this variable should be set to None when n_speakers is used instead
    :param data: DataFrame with paths and labels
    :param n_speakers_out: number of speakers to leave out per class (i.e., if n_speakers_out=2, this means that
        we leave out 2 speakers from coalition and 2 from opposition). It should be even number!
    :param label_dict: dictionary with labels
    :param data_dir: dir containing the speech files
    :return: tuple with train and test dictionaries
    """

    # pre-process
    data_pruned = copy.deepcopy(data)
    name_status = pd.crosstab(data.name, data.status)
    counts = data.value_counts(['name', 'status']).to_dict()
    if equal_sets:
        set_diff = {mp: name_status.loc[mp]['coalition'] - name_status.loc[mp]['opposition'] for mp in name_status.index
                    if not (name_status.loc[mp]['opposition'] == 0 or name_status.loc[mp]['coalition'] == 0)}
        to_remove = {
            key: ('opposition', data[(data['name'] == key) & (data['status'] == 'opposition')].index[
                                    np.random.permutation(name_status.loc[key]['opposition'])][0:abs(val)]) if np.sign(
                val) <= 0 else ('coalition', data[(data['name'] == key) & (data['status'] == 'coalition')].index[
                                                 np.random.permutation(name_status.loc[key]['coalition'])][0:abs(val)])
            for key, val in set_diff.items()
        }
        data_pruned.drop(np.hstack([list(drop_indices[1]) for drop_indices in to_remove.values()]), inplace=True)
        data_pruned.reset_index(drop=True, inplace=True)

    # compute speakers stats from coalition and opposition
    coalition_speakers = data_pruned[data_pruned['status'] == 'coalition'].reset_index(drop=True)
    opposition_speakers = data_pruned[data_pruned['status'] == 'opposition'].reset_index(drop=True)
    shared_speakers = set(opposition_speakers['name'].unique()).intersection(set(coalition_speakers['name'].unique()))
    speaker_only_in_coalition = set(coalition_speakers['name'].unique()) - shared_speakers
    speaker_only_in_opposition = set(opposition_speakers['name'].unique()) - shared_speakers
    opposition_gender_stats = opposition_speakers.value_counts('sex').to_dict()
    coalition_gender_stats = coalition_speakers.value_counts('sex').to_dict()
    max_coalition_rep = max(coalition_gender_stats, key=lambda k: coalition_gender_stats[k])
    max_opposition_rep = max(opposition_gender_stats, key=lambda k: opposition_gender_stats[k])
    coalition_reduce_by = abs(coalition_gender_stats['f'] - coalition_gender_stats['m'])
    opposition_reduce_by = abs(opposition_gender_stats['f'] - opposition_gender_stats['m'])

    # remove speakers that are either only in opposition or in coalition
    # TODO: add this here!

    if n_speakers_out is not None:
        # split speakers: we keep n_speakers from coalition and n_speakers from opposition out and keep the remaining
        # for training. For the training data, we gender-balance the number of recordings for coalition and opposition
        # to roughly have 50-50 male-female in both coalition and opposition. We also distribute equally this reduction
        # across the speakers.
        random_n_speakers = False
        if random_n_speakers:
            idx_coalition_test = np.random.permutation(len(speaker_only_in_coalition))[0:n_speakers_out]
            idx_opposition_test = np.random.permutation(len(speaker_only_in_opposition))[0:n_speakers_out]
            idx_heldout_set = np.random.permutation(len(shared_speakers))[0:2 * n_speakers_out]
            heldout_coalition = np.hstack(
                [data_pruned[(data_pruned['name'] == list(shared_speakers)[idx]) & (
                            data_pruned['status'] == 'coalition')].index.to_list() for
                 idx in
                 idx_heldout_set[0:n_speakers_out]]
            )
            heldout_opposition = np.hstack(
                [data_pruned[(data_pruned['name'] == list(shared_speakers)[idx]) & (
                            data_pruned['status'] == 'opposition')].index.to_list() for
                 idx in
                 idx_heldout_set[n_speakers_out:2 * n_speakers_out]]
            )
            training_set_speakers = set(data_pruned['name'].unique()) - set(
                [list(shared_speakers)[idx] for idx in idx_heldout_set])
            test_idx = np.hstack((heldout_opposition, heldout_coalition))
            train_idx = np.hstack(
                [data_pruned[data_pruned['name'] == speaker].index.to_list() for speaker in training_set_speakers])
        else:
            idx_start = np.random.randint(len(shared_speakers) // 2)
            utterances_per_speaker = {mp: (
                len(data_pruned[data_pruned['name'] == mp]), data_pruned[data_pruned['name'] == mp]['sex'].iloc[0],
                data_pruned[data_pruned['name'] == mp]['status'].iloc[0]) for mp in shared_speakers}
            utterances_per_speaker = sorted(utterances_per_speaker.items(), key=lambda item: item[1][0])
            heldout_idx_coalition_all = np.arange(idx_start, len(utterances_per_speaker), 2)
            heldout_idx_opposition_all = np.arange(idx_start+1, len(utterances_per_speaker), 2)
            heldout_idx_coalition = [idx for idx in heldout_idx_coalition_all if
                                     utterances_per_speaker[idx][1][1] == 'f'][
                                    0:n_speakers_out // 2] + [idx for idx in heldout_idx_coalition_all if
                                                              utterances_per_speaker[idx][1][1] == 'm'][
                                                             0:n_speakers_out // 2]
            heldout_idx_opposition = [idx for idx in heldout_idx_opposition_all if
                                      utterances_per_speaker[idx][1][1] == 'f'][
                                     0:n_speakers_out // 2] + [idx for idx in heldout_idx_opposition_all if
                                                               utterances_per_speaker[idx][1][1] == 'm'][
                                                              0:n_speakers_out // 2]
            heldout_coalition = np.hstack(
                [data_pruned[(data_pruned['name'] == utterances_per_speaker[idx][0]) & (
                        data_pruned['status'] == 'coalition')].index.to_list() for
                 idx in
                 heldout_idx_coalition]
            )
            heldout_opposition = np.hstack(
                [data_pruned[(data_pruned['name'] == utterances_per_speaker[idx][0]) & (
                        data_pruned['status'] == 'opposition')].index.to_list() for
                 idx in
                 heldout_idx_opposition]
            )
            training_set_speakers = set(data_pruned['name'].unique()) - set(
                [utterances_per_speaker[idx][0] for idx in heldout_idx_coalition + heldout_idx_opposition])
            test_idx = np.hstack((heldout_opposition, heldout_coalition))
            train_idx = np.hstack(
                [data_pruned[data_pruned['name'] == speaker].index.to_list() for speaker in training_set_speakers])
    elif named_speakers_out is not None:
        raise NotImplementedError(
            'named_speakers_out not yet implemented!')
    else:
        raise RuntimeError(
            'Either named_speakers_out or n_speakers_out should be given as an argument!')

    # get splits
    train_data = {'paths': [path if Path(path).is_absolute() else Path(data_dir).joinpath(path)
                            for path in data_pruned['path'].to_numpy()[train_idx]],
                  'labels': [label_dict[fname] for fname in
                             [Path(path).name for path in list(data_pruned['path'].to_numpy()[train_idx])]],
                  }
    test_data = {'paths': [path if Path(path).is_absolute() else Path(data_dir).joinpath(path)
                           for path in data_pruned['path'].to_numpy()[test_idx]],
                 'labels': [label_dict[fname] for fname in
                            [Path(path).name for path in list(data_pruned['path'].to_numpy()[test_idx])]],
                 }

    # add features if available
    conditional_update_dict(train_data, data, 'features', train_idx)
    conditional_update_dict(test_data, data, 'features', test_idx)

    # add vars if requested
    for label in include_labels:
        update_dict(train_data, data, label, train_idx)
        update_dict(test_data, data, label, test_idx)

    return train_data, test_data


def train() -> None:
    """
        :return:
    """
    pass


def test() -> None:
    """
        :return:
    """
    pass


def aggregate_features(features: torch.Tensor = None, method: str = 'mean', dim: int = 0) -> torch.Tensor:
    """
        :return:
    """
    if method == 'mean':
        return torch.mean(features, dim=dim).unsqueeze(0)
    elif method == 'std':
        return torch.std(features, dim=dim).unsqueeze(0)
    elif method == 'meanstd':
        return torch.hstack((torch.mean(features, dim=dim), torch.std(features, dim=dim))).unsqueeze(0)
    else:
        raise NotImplementedError(
            'Feature aggregator not available. Select between mean/std/meanstd')


def extract_features(datastruct: Dict = None, feature_set: str = 'kaldi', feature_subset: str = 'mfccs',
                         pooling_method: str = 'meanstd', truncate_signals: int = None) -> Dict:
    """
    Extract features for speech. There are two main options and several suboptions that need to be defined
    within the function definition.
    Parameters:
    -datastruct: struct containing paths and labels for the input signals
    -feature_set: opensmile/kaldi

        :return: Dictionary with processed features
    """

    features = []
    labels = []
    n = len(datastruct['paths'])
    if feature_set == 'opensmile':
        # initialize opensmile for feature extraction
        # feature levels: Functionals/LowLevelDescriptors/LowLevelDescriptors_Deltas
        # feature sets: ComParE_2016/GeMAPSv01a/GeMAPSv01b/eGeMAPSv01a/eGeMAPSv01b/eGeMAPSv02
        resample_rate = 8000  # set to None and resample to False to force implicit check of sr by opensmile routine
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,  # ComParE_2016
            feature_level=opensmile.FeatureLevel.Functionals,
            loglevel=2,
            logfile='smile.log',
            sampling_rate=resample_rate,
            resample=True,
            channels=[0],
        )

        # get feature names
        feature_names = smile.feature_names

        # process files
        for file, label in tqdm(zip(datastruct['paths'], datastruct['labels']), total=n,
                                desc="Extracting opensmile features"):
            signal, sr = audiofile.read(
                file,
                duration=truncate_signals,  # explicitly truncate signal to N secs
                offset=0,  # TODO: add random offsets
                always_2d=True,
            )
            y = smile.process_signal(signal, sr)
            features.append(y.iloc[0].values.flatten())
            labels.append(label)
    elif feature_set == 'kaldi':
        spectral_feature = feature_subset  # choose between mfccs/fbanks/spectrograms
        aggregation_method = pooling_method  # choose between mean/std/meanstd
        resample_rate = 8000
        for file, label in tqdm(zip(datastruct['paths'], datastruct['labels']), total=n,
                                desc=f'Extracting {pooling_method} {feature_subset}'):
            metadata = torchaudio.info(file)
            if metadata.num_frames != 0:
                resampler = torchaudio.transforms.Resample(metadata.sample_rate, resample_rate, dtype=torch.float32)
                signal, sr = torchaudio.load(
                    file,
                    frame_offset=0,
                    num_frames=-1 if truncate_signals is None else truncate_signals*metadata.sample_rate,
                )
                signal = resampler(signal)
                # signal, sr = audiofile.read(
                #     file,
                #     duration=truncate_signals,  # explicitly truncate signal to N secs
                #     offset=0,  # TODO: add random offsets
                #     always_2d=True,
                # )
                if spectral_feature == 'mfccs':
                    feature_id = 'C'
                    # this is the standard mfcc kaldi recipe used in ASR (the shape is (m, num_ceps))
                    y = torchaudio.compliance.kaldi.mfcc(
                        waveform=signal,  # torch.Tensor(torch.Tensor(signal).unsqueeze(0)),
                        window_type='hamming',
                        frame_length=25,
                        frame_shift=10,
                        num_ceps=13,  # we can change this to 20 or 64 for more refined representation
                        num_mel_bins=23,  # we can change this to 64 for more refined spectral representation
                        low_freq=20,
                        high_freq=3700,  # this means that we have 8kHz sr, we can adjust it (e.g. 7800 for 16kHz, etc.)
                        dither=0,  # we can increase this if we notice effects of noise flooring/gating in the signals
                        energy_floor=1.0,
                        remove_dc_offset=True,
                        subtract_mean=False,
                        htk_compat=False,
                        snip_edges=False,
                    )
                    # add deltas and deltadeltas
                    cdeltas = torchaudio.transforms.ComputeDeltas()
                    deltas = cdeltas(y)
                    deltadeltas = cdeltas(deltas)
                    y = torch.hstack((y, deltas, deltadeltas))
                    # add cepstral mean and variance normalization for utterance-level tasks
                    cmn = torchaudio.transforms.SlidingWindowCmn(
                        cmn_window=600,
                        min_cmn_window=100,
                        center=False,
                        norm_vars=False,
                    )
                    y = cmn(y)
                    y = aggregate_features(y, method=aggregation_method)
                elif spectral_feature == 'fbanks':
                    feature_id = 'FB'
                    y = torchaudio.compliance.kaldi.fbank(
                        waveform=signal,  # torch.Tensor(torch.Tensor(signal).unsqueeze(0)),
                        window_type='hamming',
                        frame_length=25,
                        frame_shift=10,
                        num_mel_bins=40,  # we can change this to 64 for more refined spectral representation
                        low_freq=20,
                        high_freq=3700,  # this means that we have 8kHz sr, we can adjust it (e.g. 7800 for 16kHz, etc.)
                        dither=0,  # we can increase this if we notice effects of noise flooring in the signals
                        energy_floor=1.0,
                        use_log_fbank=True,
                        remove_dc_offset=True,
                        subtract_mean=False,
                        htk_compat=False,
                        snip_edges=False,
                    )
                    y = aggregate_features(y, method=aggregation_method)
                elif spectral_feature == 'spectrograms':
                    feature_id = 'LOG_SPEC'
                    spect = torchaudio.transforms.Spectrogram(
                        n_fft=800,
                        win_length=25,
                        hop_length=10,
                    )
                    y = spect(signal)
                    y = aggregate_features(y, method=aggregation_method, dim=2).squeeze(1)
                else:
                    raise NotImplementedError(
                        'Spectral feature selection not available. Select between mfccs/fbanks/spectrograms')
                features.append(y)
                labels.append(label)
            else:
                warnings.warn(f'\nFile {file} had zero frames! Skipping...')
        feature_names = [
            f'{feature_id}_{idx}_{functional}' for functional in ('mean', 'std') for idx in
            np.arange(0, y.shape[1] // 2)] if aggregation_method == 'meanstd' else \
            [f'{feature_id}_{idx}_{aggregation_method}' for idx in np.arange(0, y.shape[1])]
    else:
        raise NotImplementedError('feature_set selection not available. Select between opensmile/kaldi')

    return {'features': np.vstack(features), 'labels': labels, 'feature_names': feature_names}


if __name__ == "__main__":
    main()
