# Machine Learning with Kernel Methods 2021
> [Kaggle datachallenge](https://www.kaggle.com/c/machine-learning-with-kernel-methods-2021) for the MVA course "Machine Learning with Kernel Methods"

## Problem presentation
The goal of the data challenge is to learn how to implement machine learning algorithms, gain understanding about them and adapt them to structural data.
For this reason, we have chosen a sequence classification task: predicting whether a DNA sequence region is binding site to a specific transcription factor.

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.
In this challenge, we will work with three datasets corresponding to three different TFs.

## How to run the project
### Adding the data
Please add the Kaggle data files in the `data` folder (15 files).

### Python part
In the Python part, we implemented a C-SVM with both the spectrum kernel and the substring kernel. 

In order to run the experiment, just run:
```
pip install -r requirements.txt
python run_exp.py
```

You can choose which experiment to run with the `cfg.yml` file.

### Matlab part
In the Matlab part, we implemented a C-SVM with the substring kernel. In order to run it, you can directly run the file `SVM.m`. 

As for the other files:
- `kernel_substring.m`: function to compute the Gram matrix of the substring kernel (of a given input)
- `test_kernel_substring.m`: script to run the function above
- `train.m`: function used to train the SVM on the training dataset and make predictions on the test dataset