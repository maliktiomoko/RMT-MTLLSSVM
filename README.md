# RMT-MTLLSSVM
Code for Random Matrix Improved Multi Task Learning Least Square Support Vector Machine

This document explains how to use the code implementing the Random Matrix Improved Multi-Task Learning Least Square Support Vector Machine (RMT-MTLLSSVM) proposed in the article 'Large Dimensional Analysis of Multi Task Learning'

The function implementing the multi class naive extension is called ``RMTMTLLSSVM_train.m`` which trains the MTL LS-SVM proposed algorithm.
The function implementing the binary classification is called ``MTLLSSVMTrain_binary.m`` which trains the MTL LS-SVM proposed binary algorithm.
The functions implementing a complete but slow multi class extensions ``RMTMTLLSSVM_train_one_all``, ``RMTMTLLSSVM_train_one_one``, ``RMTMTLLSSVM_train_one_hot``.
The main script comparing all algorithms for synthetic data/real data is ``CompareMTL.m`` for the naive multi class classification.
The main script comparing all algorithms for synthetic data/real data is ``CompareMTL_general.m`` for the different multi class classification.
The main script illustrating the binary classification with fixed probability of false alarm is ``PFA.m``.
Folder utils: containing alternative MTL algorithms among which MMDT algorithm, CDLS algorithm, ILS algorithm and LS-SVM on source or target and other functions used for the proposed method.
Folder datasets: containing Office+Caltech dataset Mit-Bih dataset and Mnist dataset.


# Code for binary MTL LSSVM
Algorithm 1 of the main paper training and optimizing a binary MTL LSSVM is implemented in the function ``MTLLSSVMTrain_binary.m``. As example to test the function see ``binary_experiments`` for an experiment on synthetic data (Figure 3 of the main paper). For experiments on real dataset (MNIST dataset), see script ``MNIST_experiments.m`` (for Figure 4 of the main paper) and ``more_task_experiments`` for experiments on more than one task (Figure 5 of the main paper). As general guidelines to test the code, run the algorithm for synthetic dataset (check that the theoretical predictions are close with the gaussian histogram appearing) and then test on real dataset.

# Code for multi class MTL LSSVM
For multi class classification, the code implementing respectively Algorithm 4, 5 and 6 are respectively in the functions ``RMTMTLLSSVM_train_one_all``, ``RMTMTLLSSVM_train_one_one``, ``RMTMTLLSSVM_train_one_hot``. To test the different functions, please use ``compare_MTL_general`` (to obtain as example Table~3 of the main paper). For a simple version (fast algorithm) but considering a naive multi class extension please see function ``RMTMTLLSSVM_train.m`` and to test the script ``compare_MTL`` (to reproduce Table 1 and 4).

# Code CompareMTL.m
The different options proposed to execute the script ``CompareMTL.m`` comparing the different Multi Task algorithms are as follows:
    dataset to be chosen as 'a-d' such that  a (for the source) and d (for target) are chosen between \textit{A for Amazon}, \textit{Ca for Caltech}, \textit{D for DSLR}, \textit{W for Webcam}. For illustration, for a MTL with source task Amazon and target task Caltech, the setting is ``dataset='A-Ca'``
    ``number_trials`` which represents the number of trials to be tested for each dataset in order to average performances.
    
# Code PFA.m
The different options proposed to execute the script PFA.m are as follows:
data to be chosen between 'synthetic' and \textit{'real'} to test the binary classification with fixed probability of false alarm either on synthetic data or real data.``n_training_sample`` is the number of training examples to be sampled from the overall training set.
