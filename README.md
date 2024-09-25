# 2DIR pre-trained model
This package provides a universal transfer learning model based on 2DIR spectra analysis. Having assimilated signal features from approximately 204,300 spectra, the model effectively establishes a correlation between spectra and protein conformations. It excels at accurately predicting the dynamic secondary structure content along protein folding trajectories.

![Image text](https://github.com/SaintCloud-0013/2DIR-ML/blob/main/figures/Figure1.png)

## Pre-training and fine-tuning results
We conducted molecular dynamics simulations on proteins from the [CATH database](https://www.cathdb.info/browse/tree) to construct a pre-training dataset.

![Image text](https://github.com/SaintCloud-0013/2DIR-ML/blob/main/figures/Figure2A.png)

![Image text](https://github.com/SaintCloud-0013/2DIR-ML/blob/main/figures/Figure2B.png)

The fine-tuning dataset is derived from protein folding trajectories simulated on the [Anton supercomputer](https://www.deshawresearch.com/technology.html).

![Image text](https://github.com/SaintCloud-0013/2DIR-ML/blob/main/figures/Figure3.png)

## Code usage
Please use the code [train.py](https://github.com/SaintCloud-0013/2DIR-ML/blob/main/2dir_ml/train.py) and the [pre-trained weight (Google Drive, 326 MB)](https://drive.google.com/file/d/1zpMjeEr3i9k45HjBlF6f98qBy8UuGDlF/view?usp=drive_link) to test or fine-tune your spectral data.

## Citation
If you are interested in our complete workflow and experimental methods, please refer to the following paper:
1. Fan Wu, Yan Huang, Guokun Yang, Sheng Ye, Shaul Mukamel, and Jun Jiang. Unravelling dynamic protein structures by two-dimensional infrared spectra with a pretrained machine learning model. [*PNAS* **121**, e2409257121 (2024)](https://doi.org/10.1073/pnas.2409257121).
