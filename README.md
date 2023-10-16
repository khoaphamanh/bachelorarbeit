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
## Visualize
The visualization folder is where data can be displayed. The file raw.ipynb is the raw data as a time series and is scaled.ipynb is the normalized time series. The remaining files represent each time series-to-image Encoding corresponding to each time window on the run to failure time series with window_size equal to 25600 and hop_size equal to 2560. Specifically:
- stft.ipynb: Visualize of Short-time Fourier transform.
- lms.ipynb: Visualize of the Log Mel Spectrogram.
- cwt.ipynb: Visualize of Continuous wavelet transform.
- rp.ipynb: Visualize of Recurrence plot.
- gaf.ipynb: Visualize of Gramian angular field.
- mtf.ipynb: Visualize of Markov transition field.
- helper.py: Function to read data
## Approaches
In this study, four approaches were proposed, including im_CNN, ts_im_CNN, Pann, and tcl_CNN, each in its name directory.
### im_CNN, ts_im_CNN and tcl_CNN
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
To rerun the code for training training and testing, it is necessary to run this command line in the terminal, where method.sh is short.sh, logmel.sh, conwave, recurrence.sh, gramian.sh or markov.sh.
```bash
sbatch method.sh 
```