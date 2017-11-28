# cmput551-mini-project

This repo ist for the CMPUT 551 course (Machine Learning) of the University of Alberta in the Fall 2017. The course material can be found under https://marthawhite.github.io/mlcourse/.

It tries to compare different machine learning classification algorithms for the travistorrent dataset. For a detailed description of the repository please read report/report-draft.pdf

# Usage

Tested under Ubuntu 16.04 with Python3.5.2. Python dependencies to sklearn, numpy, pymysql and prettytable.

1. Download MySQL dump from https://travistorrent.testroots.org
2. Import data into MySQL database 'archi' with user 'archi' and password 'archi'. If different names are chosen, the build_matrix.py script has to be modified.
3. Run build_matrix.py to get A.npy and y.npy (alternatively use A_small.npy and y_small.npy)
4. Run mini_project.py (This can take up to 10 hours, if not performed on a cluster)
5. Run significance.py for statistically significant results

# Results

The mini_project.py returns the best found parameters, the meaned confusion tables for the data and saves confusion.npy for the confusion table for every run for every algorithm.
This file is read by significance_py which calculates Accuracy, Sensitivity and Specificity over the ten runs and computes a two tailed t-test and interprets the results with alpha = 0.05.
