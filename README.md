This is a sample starting kit for Predicting Generalization in Deep Learning challenge at NeurIPS 2020.

Prerequisites:
- Python 3.6.6
- Tensorflow 2.0
- pandas
- pyyaml
- scikit-learn

Usage:

(1) If you are a challenge participant:

- The three files in sample_code_submission.zip are sample submissions ready to go!

- modify sample_code_submission to provide a better predictor

- zip the contents of sample_code_submission (without the directory, but with metadata), or

- to verify your code will run properly (double check you are running the correct version of python):

  `python ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`

- if you wish to test on the larger public data, download the public data and run:

  `python ingestion_program/ingestion.py **path/to/public/input_data** sample_result_submission ingestion_program sample_code_submission`

- if you wish to compute the score of your submission locally, you can run the scoring program:

  `python scoring_program/score.py **path/to/public/reference_data** **path/to/prediction** **path/to/output**`

The `baselines` directory contains a number of baselines that you may use as your starting points. If you are not familiar with the
Keras framework, these baselines should get you up to speed.

# Scores on the public set/private set

Tables generated with https://www.tablesgenerator.com/markdown_tables:

|                  Measure                  | Public | Private | Task1 Public | Task 2 Public | Task 4 Private | Task 5 Private |
|:-----------------------------------------:|:------:|:-------:|:------------:|:-------------:|:--------------:|:--------------:|
| Baseline 1 - Distance from initialization |  4.92  |   2.02  |     5.13     |      4.71     |      2.78      |      1.26      |
|           Baseline 2 - Jacobian           |  2.04  | **4.19**|     0.94     |      3.15     |      1.09      |      7.30      |
|           Baseline 3 - Sharpness          |  0.00  |   0.82  |     0.00     |      0.00     |      0.85      |      0.79      |
|           Baseline 4 - VC Dimension       |  0.02  |   0.04  |     0.00     |      0.00     |      0.85      |      0.79      |
|  Smoothness Gap (Cos 10 500-graphs-k=20)  |  14.45 |   0.72  |     9.31     |      19.58    |      0.44      |      1.00      |
|  Smoothness Last (RBF 20 500-graphs-k=50) |  9.38  |   1.38  |     8.17     |      10.58    |      1.71      |      1.06      |
|  Smoothness Gap (RBF 20 500-graphs-k=50)  |  5.47  |         |     4.95     |      6.00     |                |                |
|  Smoothness Max (Binary 1 550-graphs-k=1) |**32.6**|   0.37  |     27.74    |      37.44    |      0.21      |      0.55      |
|                Margin gap                 |  3.04  |   2.59  |     2.13     |      3.96     |      3.96      |      1.22      |
|         Input Mixup (256x256 samples)     |        |   3.63  |              |               |      0.41      |      6.85      |
|      Lipschitz Norm (128x256 samples)     |        |   4.24  |              |               |      1.25      |      7.21      |
|               One Lipschitz               |        |   4.27  |              |               |      1.23      |      7.31      |
|         Input Mixup Lipschitz Norm        |        |   4.35  |              |               |      1.18      |      7.52      |
|            SoftMax Lipschitz Norm         |        |   5.55  |              |               |      1.78      |      9.32      |
|  MIXUP Smoothness Last (RBF 100 500-graphs-k=5) |  15.6  |   5.99  |     6.31     |      24.35    |      7.88      |      4.10      |
|  MIXUP Smoothness Last (Binary 100 500-graphs-k=1) |  16.04  |   7.29  |     3.12     |      28.97    |      9.34      |      5.37      |
|  MIXUP Smoothness Penultimate (Binary 100 500-graphs-k=1 normalized) |  11.17  |   13.04  |     4.80     |      17.54    |      15.42      |      10.66      |
|      Catastrophic Forgetting Measure      |        |   0.68  |              |               |      0.61      |      0.74      |
|            Adversarial Radii              |  16.37 |   9.26  |     15.63    |     17.11     |      17.61     |      0.92      |
|         Big Adversarial Radii             |  17.80 |   8.26  |     16.31    |     19.28     |      15.82     |      0.71      |
|         Weighted Inverse Radii            |  13.99 |         |     12.84    |     15.14     |                |                |  

# Time spent on the public set/private set

|                  Time (min)               | Task1 Public | Task 2 Public | Task 4 Private | Task 5 Private |
|:-----------------------------------------:|:------------:|:-------------:|:--------------:|:--------------:|
|          *Maximum time allowed*           |    *480*     |    *270*      |      *480*     |     *320*      |
| Baseline 1 - Distance from initialization |              |               |       0.46     |     0.26       |
|           Baseline 2 - Jacobian           |              |               |       104      |     67         |
|           Baseline 3 - Sharpness          |              |               |       190      |      103       |
|           Baseline 4 - VC Dimension       |              |     0.5       |       190      |      103       |
|  Smoothness Gap (Cos 10 500-graphs-k=20)  |      6       |     3         |       15       |     10         |
|                Margin gap                 |      38      |       26      |       81       |       51       |
|         Input Mixup (256x256 samples)     |              |               |       118      |       75       |
|      Lipschitz Norm (128x256 samples)     |              |               |       193      |       120      |
|               One Lipschitz               |              |               |       346      |       217      |
|         Input Mixup Lipschitz Norm        |              |               |       90       |       56       |
|            SoftMax Lipschitz Norm         |              |               |       188      |       117      |
|  MIXUP Smoothness Last (RBF 100 500-graphs-k=5) |   472        |   110         |      143       |       103      |
|  MIXUP Smoothness Penultimate (Binary 100 500-graphs-k=1 normalized) |   468        |   109         |      142       |       100      |
|       Catastrophic Forgetting Measure     |              |               |       223      |       142      |
|            Adversarial Radii              |      54      |       32      |       43       |       38       |
|         Big Adversarial Radii             |      138     |       75      |                |                |
|        Weighted Inverse Radii             |      117     |       44      |                |                |
