import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
LITERATURE_RESULTS_FOLDER = os.path.join(BASE_FOLDER, 'real_data', 'real_data_results')
OUR_RESULTS_FOLDER = os.path.join(BASE_FOLDER, 'real_data', 'our_tree_results')

DATASETS = {
    'airline': 'airline.csv',
    'electricity': 'electricity.csv',
    'covertype': 'covertypeNorm.csv',
}


def read_result_csv(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[0] == '':
            header[0] = 'n'
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append({header[i]: float(row[i]) for i in range(len(header))})
    return header, rows


def plot_accuracy(dataset):
    literature_path = os.path.join(LITERATURE_RESULTS_FOLDER, DATASETS[dataset])
    our_path = os.path.join(OUR_RESULTS_FOLDER, dataset + '_accuracy.csv')
    lit_header, lit_rows = read_result_csv(literature_path)
    our_header, our_rows = read_result_csv(our_path)

    plt.clf()
    for column in lit_header[1:]:
        plt.plot([row['n'] for row in lit_rows], [row[column] for row in lit_rows],
                 label=column, linestyle='--')
    for column in our_header[1:]:
        plt.plot([row['n'] for row in our_rows], [row[column] for row in our_rows],
                 label=column)

    plt.xlabel('number of processed examples')
    plt.ylabel('windowed accuracy [%]')
    plt.legend()
    os.makedirs(OUR_RESULTS_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(OUR_RESULTS_FOLDER, dataset + '_accuracy_comparison.png'))


def plot_single_result(dataset, suffix, ylabel):
    path = os.path.join(OUR_RESULTS_FOLDER, dataset + '_' + suffix + '.csv')
    header, rows = read_result_csv(path)
    plt.clf()
    for column in header[1:]:
        plt.plot([row['n'] for row in rows], [row[column] for row in rows], label=column)
    plt.xlabel('number of processed examples')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(OUR_RESULTS_FOLDER, dataset + '_' + suffix + '.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    args = parser.parse_args()

    plot_accuracy(args.dataset)
    plot_single_result(args.dataset, 'memory_mb', 'tree memory [MB]')
    plot_single_result(args.dataset, 'leaves', 'number of leaves')


if __name__ == '__main__':
    main()
