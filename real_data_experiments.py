import argparse
import csv
import os
import pickle
from multiprocessing import Pool

import numpy as np

from Attribute import Attribute
from Criterion import Criterion
from Statistics import Statistics
from Tools import Tools
from Tree import Tree


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_FOLDER = os.path.join(BASE_FOLDER, 'real_data', 'real_data_data')
REAL_RESULTS_FOLDER = os.path.join(BASE_FOLDER, 'real_data', 'our_tree_results')


DATASETS = {
    'airline': 'airline.arff',
    'electricity': 'electricity.arff',
    'covertype': 'covtypeNorm.arff',
}


def parse_attribute(line):
    _, rest = line.split(None, 1)
    name, type_spec = rest.strip().split(None, 1)
    name = name.strip("'\"")
    type_spec = type_spec.strip()
    if type_spec.lower() in ['real', 'numeric', 'integer']:
        return name, 'numerical', None
    if type_spec.startswith('{') and type_spec.endswith('}'):
        values = [value.strip() for value in type_spec[1:-1].split(',')]
        return name, 'nominal', values
    raise ValueError('Unsupported ARFF attribute type: ' + line)


def read_arff_schema(path):
    attributes = []
    with open(path, encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@attribute'):
                attributes.append(parse_attribute(line))
            elif line.lower().startswith('@data'):
                break
    if len(attributes) < 2:
        raise ValueError('ARFF file has too few attributes: ' + path)
    return attributes[:-1], attributes[-1]


def iter_arff_rows(path):
    in_data = False
    with open(path, encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('%'):
                continue
            if not in_data:
                if line.lower().startswith('@data'):
                    in_data = True
                continue
            if line.startswith('{'):
                raise ValueError('Sparse ARFF rows are not supported')
            yield [value.strip() for value in next(csv.reader([line]))]


def build_attributes(path, nbins):
    attribute_specs, class_spec = read_arff_schema(path)
    mins = [None for _ in attribute_specs]
    maxs = [None for _ in attribute_specs]

    for row in iter_arff_rows(path):
        for idx, (name, attr_type, values) in enumerate(attribute_specs):
            if attr_type != 'numerical' or row[idx] == '?':
                continue
            value = float(row[idx])
            mins[idx] = value if mins[idx] is None else min(mins[idx], value)
            maxs[idx] = value if maxs[idx] is None else max(maxs[idx], value)

    attributes = []
    for idx, (name, attr_type, values) in enumerate(attribute_specs):
        if attr_type == 'numerical':
            if mins[idx] is None or maxs[idx] is None:
                raise ValueError('No numeric values found for attribute ' + name)
            if mins[idx] == maxs[idx]:
                maxs[idx] = mins[idx] + 1.0
            attributes.append(Attribute('numerical', range_a=mins[idx], range_b=maxs[idx], nbins=nbins))
        else:
            attributes.append(Attribute('nominal', values))

    class_name, class_type, class_values = class_spec
    if class_type != 'nominal':
        raise ValueError('Only nominal class attributes are supported')
    return attributes, Attribute('nominal', class_values)


def create_trees(dataset_name, attributes, classes, fractions, preq_lambda, window_size):
    statistics = Statistics(attributes, classes)
    trees = []
    for fraction in fractions:
        criterion = Criterion(classes.getN(), 0.001, 'hoeffding', 'gini', fraction, False, 0.0)
        suffix = str(fraction).replace('.', 'p')
        trees.append((f'f={fraction}', Tree(f'{dataset_name}_f{suffix}', statistics, [criterion], False,
                                            preq_lambda, window_size)))
        trees.append((f'f={fraction}_ES', Tree(f'{dataset_name}_f{suffix}_ES', statistics, [criterion], True,
                                               preq_lambda, window_size)))
    return trees


def create_tree(dataset_name, attributes, classes, fraction, extended_statistics, preq_lambda, window_size):
    statistics = Statistics(attributes, classes)
    criterion = Criterion(classes.getN(), 0.001, 'hoeffding', 'gini', fraction, False, 0.0)
    suffix = str(fraction).replace('.', 'p')
    name = f'f={fraction}'
    tree_name = f'{dataset_name}_f{suffix}'
    if extended_statistics:
        name = name + '_ES'
        tree_name = tree_name + '_ES'
    return name, Tree(tree_name, statistics, [criterion], extended_statistics, preq_lambda, window_size)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_tree_without_sequence(tree, data_vector):
    ignored_results, split_attr = tree.root.pass_data(data_vector, tree.criteria_list, [0], True)
    if split_attr > -1:
        tree.leaves.append(tree.leaves[-1] + 1)
    else:
        tree.leaves.append(tree.leaves[-1])


def run_variant(args):
    dataset_name, attributes, classes, fraction, extended_statistics, window_size, preq_lambda, limit, save_models = args
    filename = DATASETS[dataset_name]
    dataset_path = os.path.join(REAL_DATA_FOLDER, filename)
    name, tree = create_tree(dataset_name, attributes, classes, fraction, extended_statistics, preq_lambda, window_size)
    correct_in_window = 0
    accuracy = []
    memory_mb = []
    leaves = []

    for n, raw_row in enumerate(iter_arff_rows(dataset_path), start=1):
        if limit is not None and n > limit:
            break
        if '?' in raw_row:
            continue
        data_vector = Tools.transform_data_vector(raw_row, attributes, classes)
        prediction, ignored_sa = tree.root.pass_data(data_vector, tree.criteria_list, [0], False)
        correct_in_window = correct_in_window + int(prediction[0] == data_vector[-1])
        train_tree_without_sequence(tree, data_vector)

        if n % window_size == 0:
            accuracy.append((n, 100.0 * correct_in_window / window_size))
            memory_mb.append((n, tree.memory_size() / (1024.0 * 1024.0)))
            leaves.append((n, tree.leaves[-1]))
            correct_in_window = 0
            print(dataset_name, name, n)

    if save_models:
        os.makedirs(os.path.join(REAL_RESULTS_FOLDER, 'pkl'), exist_ok=True)
        safe_name = name.replace('=', '').replace('.', 'p')
        with open(os.path.join(REAL_RESULTS_FOLDER, 'pkl', dataset_name + '_' + safe_name + '.pkl'), 'wb') as f:
            pickle.dump(tree, f)

    return name, accuracy, memory_mb, leaves


def rows_from_variant_results(variant_results, value_idx):
    series_idx = value_idx + 1
    ns = [n for n, value in variant_results[0][series_idx]]
    rows = []
    for row_idx, n in enumerate(ns):
        row = {'n': n}
        for name, accuracy, memory_mb, leaves in variant_results:
            series = [accuracy, memory_mb, leaves][value_idx]
            row[name] = series[row_idx][1]
        rows.append(row)
    return rows


def write_variant_results(dataset_name, variant_results):
    columns = ['n'] + [name for name, accuracy, memory_mb, leaves in variant_results]
    prefix = os.path.join(REAL_RESULTS_FOLDER, dataset_name)
    write_csv(prefix + '_accuracy.csv', rows_from_variant_results(variant_results, 0), columns)
    write_csv(prefix + '_memory_mb.csv', rows_from_variant_results(variant_results, 1), columns)
    write_csv(prefix + '_leaves.csv', rows_from_variant_results(variant_results, 2), columns)


def run_dataset_parallel(dataset_name, nbins, fractions, window_size, preq_lambda, limit, save_models, jobs):
    filename = DATASETS[dataset_name]
    dataset_path = os.path.join(REAL_DATA_FOLDER, filename)
    print('preparing schema:', dataset_path)
    attributes, classes = build_attributes(dataset_path, nbins)
    tasks = []
    for fraction in fractions:
        tasks.append((dataset_name, attributes, classes, fraction, False, window_size, preq_lambda, limit, save_models))
        tasks.append((dataset_name, attributes, classes, fraction, True, window_size, preq_lambda, limit, save_models))
    with Pool(processes=jobs) as pool:
        variant_results = pool.map(run_variant, tasks)
    write_variant_results(dataset_name, variant_results)


def run_dataset(dataset_name, nbins, fractions, window_size, preq_lambda, limit=None, save_models=False, jobs=1):
    if jobs > 1:
        run_dataset_parallel(dataset_name, nbins, fractions, window_size, preq_lambda, limit, save_models, jobs)
        return

    filename = DATASETS[dataset_name]
    dataset_path = os.path.join(REAL_DATA_FOLDER, filename)
    print('preparing schema:', dataset_path)
    attributes, classes = build_attributes(dataset_path, nbins)
    trees = create_trees(dataset_name, attributes, classes, fractions, preq_lambda, window_size)

    correct_in_window = {name: 0 for name, tree in trees}
    accuracy_rows = []
    memory_rows = []
    leaves_rows = []

    for n, raw_row in enumerate(iter_arff_rows(dataset_path), start=1):
        if limit is not None and n > limit:
            break
        if '?' in raw_row:
            continue
        data_vector = Tools.transform_data_vector(raw_row, attributes, classes)

        for name, tree in trees:
            prediction, ignored_sa = tree.root.pass_data(data_vector, tree.criteria_list, [0], False)
            correct_in_window[name] = correct_in_window[name] + int(prediction[0] == data_vector[-1])
            train_tree_without_sequence(tree, data_vector)

        if n % window_size == 0:
            accuracy_row = {'n': n}
            memory_row = {'n': n}
            leaves_row = {'n': n}
            for name, tree in trees:
                accuracy_row[name] = 100.0 * correct_in_window[name] / window_size
                memory_row[name] = tree.memory_size() / (1024.0 * 1024.0)
                leaves_row[name] = tree.leaves[-1]
                correct_in_window[name] = 0
            accuracy_rows.append(accuracy_row)
            memory_rows.append(memory_row)
            leaves_rows.append(leaves_row)
            print(dataset_name, n)

    columns = ['n'] + [name for name, tree in trees]
    prefix = os.path.join(REAL_RESULTS_FOLDER, dataset_name)
    write_csv(prefix + '_accuracy.csv', accuracy_rows, columns)
    write_csv(prefix + '_memory_mb.csv', memory_rows, columns)
    write_csv(prefix + '_leaves.csv', leaves_rows, columns)

    if save_models:
        os.makedirs(os.path.join(REAL_RESULTS_FOLDER, 'pkl'), exist_ok=True)
        for name, tree in trees:
            safe_name = name.replace('=', '').replace('.', 'p')
            with open(os.path.join(REAL_RESULTS_FOLDER, 'pkl', dataset_name + '_' + safe_name + '.pkl'), 'wb') as f:
                pickle.dump(tree, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--nbins', type=int, default=10)
    parser.add_argument('--fractions', nargs='+', type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--window-size', type=int, default=200)
    parser.add_argument('--preq-lambda', type=float, default=0.999)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()

    run_dataset(args.dataset, args.nbins, args.fractions, args.window_size, args.preq_lambda, args.limit,
                args.save_models, args.jobs)


if __name__ == '__main__':
    main()
