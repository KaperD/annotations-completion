import os
import json
from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from utils.encoder import encode_column

target_types = [
    'AnnotationType',
    'Constructor',
    'Field',
    'LocalVariable',
    'Method',
    'Module',
    'Package',
    'Parameter',
    'RecordComponent',
    'Type',
    'TypeParameter',
    'TypeUse'
]

initial_feature_names = [
    'targetName',
    'className',
    'targetType',
    'modifiers',
    'otherAnnotations',
    'otherMethodsNames',
    'otherMethodsAnnotations',
    'returnsNull',
    'checksNull',
    'fileName',
    'otherParamsNames',
    'otherParamsAnnotations',
    'target'
]


class AnnotationUsage:
    def __init__(self, usage_json):
        self.annotation_name = usage_json['name']
        features_json = usage_json['features']
        self.features_list = [
            features_json.get('targetName', ''),
            features_json.get('className', ''),
            features_json.get('targetType', ''),
            features_json.get('modifiers', []),
            features_json.get('otherAnnotations', []),
            features_json.get('otherMethodsNames', []),
            features_json.get('otherMethodsAnnotations', []),
            1 if features_json.get('returnsNull', False) else 0,
            1 if features_json.get('checksNull', False) else 0,
            features_json.get('fileName', ''),
            features_json.get('otherParamsNames', []),
            features_json.get('otherParamsAnnotations', []),
        ]
        self.file_path = usage_json['filePath']

    def __str__(self):
        return f'{self.annotation_name}'


def load(usages,
         max_new_columns=100,
         size=10000,
         train_fraction=0.8,
         state=42,
         need_polynomial=False):
    usages = shuffle(usages, random_state=state)[:size]
    raw_X = np.array([np.array(usage.features_list, dtype=object) for usage in usages])
    X = None
    all_new_names = []
    if len(raw_X) == 0:
        X = np.array([])
        all_new_names = initial_feature_names
    else:
        for col in range(raw_X.shape[1]):
            new_columns, new_names = encode_column(raw_X[:, col], round(len(raw_X[:, col]) * train_fraction),
                                                   initial_feature_names[col], max_new_columns)
            if new_columns is None:
                continue
            all_new_names += new_names
            if X is None:
                X = new_columns
            else:
                X = np.concatenate((X, new_columns), axis=1)
    if need_polynomial and len(X) > 0:
        sel = VarianceThreshold(threshold=0.01)
        print(len(X[0]))
        X = sel.fit_transform(X)
        print(len(X[0]))
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X = poly.fit_transform(X)
        print(len(X[0]))
        sel = VarianceThreshold(threshold=0.05)
        X = sel.fit_transform(X)
        print(len(X[0]))
    y = np.array([usage.annotation_name for usage in usages])
    return X, y, all_new_names


class UsagesLoader:
    def __init__(self, processing_result_paths):
        usages_by_target = defaultdict(list)
        for path in processing_result_paths:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if not file.endswith('json'):
                        continue
                    with open(os.path.join(root, file), 'r') as read_file:
                        data = json.load(read_file)
                        target_type = data['keyInfo']['name']
                        new_usages = [AnnotationUsage(usage_json) for usage_json in data["usages"]]
                        for usage in new_usages:
                            usage.features_list.append(target_type)
                        usages_by_target[target_type] = usages_by_target[target_type] + new_usages
        self.usages_by_target = usages_by_target

    def load_all(self):
        all_usages = []
        for _, value in self.usages_by_target.items():
            all_usages += value
        return all_usages

    def load(self,
             max_new_columns=100,
             size=10000,
             train_fraction=0.8,
             state=42,
             need_polynomial=False,
             ignored_annotation=()):
        usages = list(filter(lambda x: x.annotation_name not in ignored_annotation, self.load_all()))
        return load(usages, max_new_columns, size, train_fraction, state, need_polynomial)

    def load_for_target(self,
                        target_type,
                        max_new_columns=100,
                        size=10000,
                        train_fraction=0.8,
                        state=42,
                        need_polynomial=False,
                        ignored_annotation=()):
        usages = list(
            filter(lambda x: x.annotation_name not in ignored_annotation, self.usages_by_target[target_type]))
        return load(usages, max_new_columns, size, train_fraction, state, need_polynomial)
