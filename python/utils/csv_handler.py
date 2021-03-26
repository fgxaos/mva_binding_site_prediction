### LIBRARIES ###
import os
import csv
import numpy as np

### FUNCTION DEFINITIONS ###
def load_training_data(list_files):
    """Loads the training datasets.

    Args:
        list_files: List[Str]
                    paths of the CSV files to load
    Returns:
        training_data: List[Str]
                       list with the elements of the training dataset
    """
    training_data = []
    for tr_file in list_files:
        with open(os.path.join("data", tr_file)) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            next(reader)
            for row in reader:
                training_data.append(row[1])
    return training_data


def load_dataset(path):
    """Loads a single training dataset.

    Args:
        file_path: str
                   path of the CSV file to load
    Returns:
        data: List[Str]
              list of the elements of the training dataset
    """
    training_data = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        next(reader)
        for row in reader:
            training_data.append(row[1])
    return training_data


def load_training_gt(list_files):
    """Loads the training ground-truth results.

    Args:
        list_files: List[Str]
                    paths of the CSV files to load
    Returns:
        training_results: np.array({0, 1})
                          array with the ground-truth results
    """
    training_results = []
    for res_file in list_files:
        with open(os.path.join("data", res_file)) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            next(reader)
            for row in reader:
                training_results.append(int(row[1]))
    return np.array(training_results)


def load_gt(path):
    """Loads the training ground-truth results.

    Args:
        path: str
              path of the CSV file to load
    Returns:
        training_results: np.array in {0, 1}
                          array with the ground-truth results
    """
    train_results = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        next(reader)
        for row in reader:
            train_results.append(int(row[1]))
    return np.array(train_results)


def write_results(file_path, predictions):
    """Writes the predicted bounds in a CSV file.

    Args:
        file_path:   Str
                     path of the file to write
        predictions: np.array({0, 1})
                     array with the predicted results
    """
    with open(file_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["Id", "Bound"])
        for id, bound in enumerate(predictions):
            writer.writerow([id, bound])
