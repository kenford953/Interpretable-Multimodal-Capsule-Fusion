# Interpretable-Multimodal-Capsule-Fusion
Code for Interpretable Multimodal Capsule Fusion, accepted by IEEE Transaction on Audio, Speech and Language Processing (TASLP), https://ieeexplore.ieee.org/abstract/document/9783105.
This repository includes data, code for the TASLP 2022 paper, "Interpretable Multimodal Capsule Fusion".

# Data
Data files ("mosei_senti_data_noalign.pkl" and "mosi_data_noalign.pkl") can be downloaded from [here](https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0). Place the data files in an appropriate folder which is described in the argument '--data-path' in 'main_grid_search.py'.

To retrieve the meta information and the raw data, please refer to the [SDK for these datasets](https://github.com/A2Zadeh/CMU-MultimodalSDK).

## Run the Code
### Requirements
- Python 3.7
- Pytorch 1.5.0
- numpy 1.20.3
- sklearn

### Train and test
~~~~
python main_grid_search.py [--FLAGS]
~~~~
It will train and test the model with different hyperparameter settings. The experimental results for different settings and the best setting will be printed in the terminal. The pretrained model for each setting will be saved in './pretrained_models'. Note that we train our model for each emotion in iemocap dataset. When you implement the experiment for iemocap, you need to modify the arguments '--dataset' and '--emotion'.
