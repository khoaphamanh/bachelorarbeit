# FEMTO BEARING
## Installation
To run the code in this project, Anaconda 23.1.0 and Python 3.10.10 must be installed. Libraries and frameworks can be installed in a virtual environment through the following commands in the terminal:
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
- gaf.ipynb: Visualize of Gramian angular fields.
- mtf.ipynb: Visualize of Markov transition field.
- helper.py: Function to read data.
## Approaches
In this study, four approaches were proposed, including im_CNN, ts_CNN, Pann, and tcl_CNN. In each folder of the same name there will be the following files with the functions listed:
- Short-time Fourier transform
    - Short-time Fourier transform
    - stft_cv.py: The time series will be converted to images using the Short-time Fourier transform method and will become the input for the model. This file includes cross validation, training on the training data set and testing on the test data set.
    - short.sh: Job script to Slurm to run stft_cv.py.
    - STFT_bash.sh: Bash shell script to run multiple short.sh one after another. Each run represents a trial in the parameter optimization process.
### im_CNN
