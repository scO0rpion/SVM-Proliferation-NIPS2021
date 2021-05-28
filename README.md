# SVM-Proliferation-NIPS2021
## Support vector machines and linear regression onlycoincide with very high-dimensional features

This repository is the codebase for experiment section of --ref-- paper. In order to reproduce our figures, we have included the datasets which we used for the analyses. 

**Generate Datasets:** Python files can be used to generate the datasets. Note that our code has the flexibility to run in parallel; number of cores can be specified, if not the code will run in a serialized fashion. The seed is not fixed in our code.

| File        | Syntax                          |
|-------------|---------------------------------|
| `l1_svm.py` | `python l1_svm.py <num_cores>`  |
| `l2_svm.py` | `python l2_svm.py <num_cores>`  |

**Analyses:** The analysis is done in R where the R-markdown file is provided. 
