# FEMTO BEARING
## Installation
To run the code in this project, Anaconda 23.1.0 and Python 3.10.10 must be installed. Libraries and frameworks can be installed in a virtual environment through the following commands in the terminal:
```bash
conda create --name your_vitual_enviroment python=3.10.10
```
```bash
source /path/to/anaconda3/bin/activate /path/to/anaconda3/envs/your_vitual_enviroment
```
```bash
conda install --file requirements.txt
```
## Download datasets
Data can be downloaded at this [link](https://seafile.cloud.uni-hannover.de/d/18bc6da305bd46fca62e/) and should be located right in this directory, only if you pull the code from gitea.

Otherwise you can download the entire work including code, data and pretrained model via the [link](https://seafile.cloud.uni-hannover.de/d/2d9dec930be54e4b9ba5/).

Each folder containing data will be annotated as follows:
- data_2560: Folder containing time series cut into each time window with *window_size* equal to 2560 and *hop_size* equal to 2560.
- data_25600: Folder containing time series cut into each time window with *window_size* equal to 25600 and *hop_size* equal to 2560.
- data_tcl: Folder containing time series cut into each time window with *window_size* equal to 25600 and *hop_size* equal to 25600.
- train: Run to failure time series of all train bearings.
- full: Run to failure time series of all test bearings.
- test: Time series has been randomly cut off in the middle of all test bearings.
- windowing: Folder containing files that read data from the train and full folders to create time windows in the data_2560, data_25600 and data_tcl folders.
## Visualize
The visualization folder is where data can be displayed. The file raw.ipynb is the raw data as a time series and is scaled.ipynb is the normalized time series. The remaining files represent each time series-to-image Encoding corresponding to each time window on the run to failure time series with *window_size* equal to 25600 and *hop_size* equal to 2560. Specifically:
- stft.ipynb: Visualize of Short-time Fourier transform.
- lms.ipynb: Visualize of the Log Mel Spectrogram.
- cwt.ipynb: Visualize of Continuous wavelet transform.
- rp.ipynb: Visualize of Recurrence plot.
- gaf.ipynb: Visualize of Gramian angular field.
- mtf.ipynb: Visualize of Markov transition field.
- helper.py: Function to read data
## Approaches
In this study, four approaches were proposed, including im_CNN, ts_im_CNN, pann, and tcl_im_CNN, each in its name directory.
### im_CNN, ts_im_CNN and tcl_im_CNN
In each folder of the approaches there will be the following files with the functions listed:
- Short-time Fourier transform
    - stft_cv.py: The time series will be converted to images using the Short-time Fourier transform method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    - short.sh: Job script to Slurm to run stft_cv.py. Each run represents a trial in the hyperparameter tunning process.
    - STFT_bash.sh: Bash shell script to run multiple short.sh one after another. 
- Log Mel Spectrogram
    - lms_cv.py: The time series will be converted to images using the Log Mel Spectrogram method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    - logmel.sh: Job script to Slurm to run lms_cv.py. Each run represents a trial in the hyperparameter tunning process.
    - LMS_bash.sh: Bash shell script to run multiple logmel.sh one after another. 
- Continuous wavelet transform
    - cwt_cv.py: The time series will be converted to images using the Continuous wavelet transform method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    - conwave.sh: Job script to Slurm to run cwt_cv.py. Each run represents a trial in the hyperparameter tunning process.
    - CWT_bash.sh: Bash shell script to run multiple conwave.sh one after another. 
- Recurrence plot
    - rp_cv.py: The time series will be converted to images using the Recurrence plot method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    - recurrence.sh: Job script to Slurm to run rp_cv.py. Each run represents a trial in the hyperparameter tunning process.
    - RP_bash.sh: Bash shell script to run multiple recurrence.sh one after another. 
- Gramian angular field
    - gaf_cv.py: The time series will be converted to images using the Gramian angular fields method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    - gramian.sh: Job script to Slurm to run gaf_cv.py. Each run represents a trial in the hyperparameter tunning process.
    - GAF_bash.sh: Bash shell script to run multiple gramian.sh one after another. 
- Markov transition field
    - mtf_cv.py: The time series will be converted to images using the Markov transition field method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    - markov.sh: Job script to Slurm to run mtf_cv.py. Each run represents a trial in the hyperparameter tunning process.
    - MTF_bash.sh: Bash shell script to run multiple markov.sh one after another. 
- tranformation.py: Funtion to transform time series to images and build the model
- model_pretrained: Directory to store the trained model
- result: Directory to store the result of the cross validation, training and testing processes

To rerun the code for training training and testing, it is necessary to run this command line in the terminal, where *method.sh* could be short.sh, logmel.sh, conwave, recurrence.sh, gramian.sh or markov.sh depends on the method of converting time series to images. _approach_ represents the approach, which can be im_CNN, ts_im_CNN or tcl_CNN.
```bash
cd path/to/bachelorarbeit/approach/
```
```bash
sbatch method.sh 
```
If you want to run code for both training and testing cross validation, the model_pretrained folder needs to be deleted and run this command in terminal, where *METHOD_bash.sh* can be STFT_bash.sh, LMS_bash.sh, CWT_bash.sh, RP_bash.sh, GAF_bash.sh or MTF_bash.sh depends on the method of converting time series to images.
```bash
cd path/to/bachelorarbeit/approach/
```
```bash
bash METHOD_bash.sh 
```
### pann
In this folder there will be the following files with the functions listed:
- pann.py: The time series will be converted to images using the Log Mel Spectrogram in model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
- pnn.sh: Job script to Slurm to run pann.py. Each run represents a trial in the hyperparameter tunning process.
- PANN.sh: Bash shell script to run multiple pnn.sh one after another. 

Same as above, to rerun the code for training training and testing, you need to enter the following command line into the terminal:
```bash
cd path/to/bachelorarbeit/pann/
```
```bash
sbatch pnn.sh 
```
If you want to run code for both training and testing cross validation, the model_pretrained folder needs to be deleted and run this command in terminal:
```bash
cd path/to/bachelorarbeit/pann/
```
```bash
bash PANN.sh 
```
## Experiment tracking
If you want to have a closer look at each trial in each method as well as charts of metrics, you can access this project's experiment tracking through [neptune](https://ui.neptune.ai/auth/realms/neptune/protocol/openid-connect/auth?client_id=neptune-frontend&redirect_uri=https%3A%2F%2Fapp.neptune.ai&state=82f29ed0-7cf1-4cc1-b98e-6f3b9e664623&response_mode=fragment&response_type=code&scope=openid&nonce=bf90715e-8faf-4544-8f0c-5193e1388d9d) by logging in to account:

Username: anh-khoa

Password: khoa9898

Select workspace ba_final in the upper left corner of the screen. Projects named stft-1, lms-1, cwt-1, rp-1, gaf-1, mtf-1 belong to the im_CNN approach, while projects stft-2, lms-2, cwt-2, rp- 2, gaf-2, mtf-2 are ts_im_CNN approach and projects stft-3, lms-3, cwt-3, rp-3, gaf-3, mtf-3 come from tcl_im_CNN approach. Finally pann-2 belongs to the pann approach. 