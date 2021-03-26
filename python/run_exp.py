### LIBRARIES ###
# Global libraries
import os
import itertools
import csv
import yaml
import wandb

import numpy as np

# Custom libraries
from svm import SVM
from utils.csv_handler import load_dataset, load_gt, write_results
from utils.utils_functions import compare, init_wandb

### LOAD CONFIGURATION ###
with open("cfg.yml", "r") as yml_file:
    cfg = yaml.safe_load(yml_file)

### SET RANDOM SEED ###
np.random.seed(cfg["np_seed"])

### CREATE FOLDERS IF THEY DON'T EXIST ###
if not os.path.exists(cfg["paths"]["data"]):
    os.makedirs(cfg["paths"]["data"])
if not os.path.exists(cfg["paths"]["k"]):
    os.makedirs(cfg["paths"]["k"])

### MAIN FUNCTION ###
# Load the different datasets
training_files = ["Xtr{}.csv".format(i) for i in range(0, 3)]
result_files = ["Ytr{}.csv".format(i) for i in range(0, 3)]
test_files = ["Xte{}.csv".format(i) for i in range(0, 3)]

# Load the training data
train_datasets = {}
for i, tr_file in enumerate(training_files):
    train_datasets["tr{}".format(i)] = load_dataset(
        os.path.join(cfg["paths"]["data"], tr_file)
    )

# Load the training targets
train_results = {}
for i, res_file in enumerate(result_files):
    train_results["tr{}".format(i)] = load_gt(
        os.path.join(cfg["paths"]["data"], res_file)
    )

# Load the test data
test_datasets = {}
for i, te_file in enumerate(test_files):
    test_datasets["te{}".format(i)] = load_dataset(
        os.path.join(cfg["paths"]["data"], te_file)
    )

for (lambd, lambd_kern, subseq_length) in itertools.product(
    cfg["lambd"], cfg["lambd_kern"], cfg["subseq_length"]
):
    train_split = cfg["train_split"]
    if cfg["use_wandb"]:
        init_wandb(lambd, lambd_kern, subseq_length, train_split, cfg["kernel"])

    print(
        "===== New experience: lambda = {}, lambd_kern = {}, train_split = {}, subseq_length = {} =====".format(
            lambd, lambd_kern, train_split, subseq_length
        )
    )

    # Loop over each training file
    test_predictions = {}
    test_length = 0
    total_svm_acc = 0
    for i, (dataset, train_data) in enumerate(train_datasets.items()):
        print(
            "=== Working on dataset tr{} ({}/{}) ===".format(
                i, i + 1, len(training_files)
            )
        )

        # Initialize the SVM
        svm = SVM(
            lambd=lambd,
            lambd_kern=lambd_kern,
            subseq_length=subseq_length,
            kernel=cfg["kernel"],
            parallel=cfg["parallel"],
        )

        if cfg["kernel"] == "spectrum":
            filename = "svm_data_{}-sub_{}.npy".format(dataset, subseq_length)
            k_file_path = os.path.join(cfg["paths"]["k"], filename)

        elif cfg["kernel"] == "substring" or cfg["kernel"] == "fast_substring":
            filename = "svm_data_{}-sub_{}-lam_{}".format(
                dataset, subseq_length, lambd_kern
            )
            k_file_path = os.path.join(cfg["paths"]["k"], filename)

        split_idx = int(len(train_data) * train_split)

        if train_split == 1:
            # Train on the whole training dataset, no validation,
            # and write the predictions on the test dataset

            # Fit the SVM object
            svm.fit(train_data, train_results["tr{}".format(i)], 0, k_file_path)

            # Predict with the SVM object on the test dataset
            test_predictions[i] = svm.predict(test_datasets["te{}".format(i)])
            test_length += len(test_predictions[i])

        else:
            # Train and validate on subsamples of the training dataset,
            # and write the predictions on the test dataset

            # Fit the SVM object
            svm.fit(train_data, train_results["tr{}".format(i)], split_idx, k_file_path)

            # Predict with the SVM object on the validation dataset
            print("Computing the accuracy with the SVM method...")
            svm_val_pred = svm.predict(train_data[split_idx:])
            svm_val_acc = compare(
                svm_val_pred, train_results["tr{}".format(i)][split_idx:]
            )
            if cfg["use_wandb"]:
                wandb.log({"svm_val_acc_{}".format(dataset): svm_val_acc})
                total_svm_acc += svm_val_acc
            else:
                print("SVM validation accuracy: ", svm_val_acc)

            # Predict with the SVM object on the test dataset
            print("Computing the predictions on the associated test dataset...")
            test_predictions[i] = svm.predict(test_datasets["te{}".format(i)])
            test_length += len(test_predictions[i])

    print("=============")
    # Concatenate the test predictions
    global_test_pred = np.zeros(test_length, dtype=int)
    start_idx = 0
    for _, test_pred in test_predictions.items():
        end_split = start_idx + len(test_pred)
        global_test_pred[start_idx:end_split] = test_pred
        start_idx += len(test_pred)

    # Save the predictions in a file
    file_pred_name = "{}_{}_{}.csv".format(cfg["test_pred_name"], lambd, subseq_length)
    write_results(file_pred_name, global_test_pred)
