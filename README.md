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
- stft.ipynb: Visualize Short-time Fourier transform.
- lms.ipynb: Visualize the Log Mel Spectrogram.
- cwt.ipynb: Visualize Continuous wavelet transform.
- rp.ipynb: Visualize Recurrence plot.
- gaf.ipynb: Visualize Gramian angular fields.
- mtf.ipynb: Visualize Markov transition field.
## Approaches
In this work, four approaches have been proposed, which include:
### im_CNN
