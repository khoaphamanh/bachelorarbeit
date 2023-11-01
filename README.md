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

Otherwise you can download the entire work including code, data and pretrained model via the [link](https://seafile.cloud.uni-hannover.de/f/4b7b2f4d5eb846fd9502/?dl=1). All this data takes up about 30 GB on the hard drive.

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
    + stft_cv.py: The time series will be converted to images using the Short-time Fourier transform method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    + short.sh: Job script to Slurm to run stft_cv.py. Each run represents a trial in the hyperparameter tunning process.
    + STFT_bash.sh: Bash shell script to run multiple short.sh one after another. 
- Log Mel Spectrogram
    + lms_cv.py: The time series will be converted to images using the Log Mel Spectrogram method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    + logmel.sh: Job script to Slurm to run lms_cv.py. Each run represents a trial in the hyperparameter tunning process.
    + LMS_bash.sh: Bash shell script to run multiple logmel.sh one after another. 
- Continuous wavelet transform
    + cwt_cv.py: The time series will be converted to images using the Continuous wavelet transform method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    + conwave.sh: Job script to Slurm to run cwt_cv.py. Each run represents a trial in the hyperparameter tunning process.
    + CWT_bash.sh: Bash shell script to run multiple conwave.sh one after another. 
- Recurrence plot
    + rp_cv.py: The time series will be converted to images using the Recurrence plot method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    + recurrence.sh: Job script to Slurm to run rp_cv.py. Each run represents a trial in the hyperparameter tunning process.
    + RP_bash.sh: Bash shell script to run multiple recurrence.sh one after another. 
- Gramian angular field
    + gaf_cv.py: The time series will be converted to images using the Gramian angular fields method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    + gramian.sh: Job script to Slurm to run gaf_cv.py. Each run represents a trial in the hyperparameter tunning process.
    + GAF_bash.sh: Bash shell script to run multiple gramian.sh one after another. 
- Markov transition field
    + mtf_cv.py: The time series will be converted to images using the Markov transition field method and will become the input for the model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
    + markov.sh: Job script to Slurm to run mtf_cv.py. Each run represents a trial in the hyperparameter tunning process.
    + MTF_bash.sh: Bash shell script to run multiple markov.sh one after another. 
- tranformation.py: Funtion to transform time series to images and build the model
- model_pretrained: Directory to store the trained model
- result: Directory to store the result of the cross validation, training and testing processes

For convenience of testers who want to run the code and compare with our results, can use arguments:
- "-d" represents the name of the directory where you want to save the pretrained models, the database file that stores trials and scalers. The default value of this argument and also the default directory we use in this work is "model_pretrained". This folder contains:
    + METHOD.db is a database file containing trials including hyperparameters tested during cross validation and their results. METHOD is the abbreviation for the time sereries-to-image methods used in this work, for example STFT.db.
    + t_i_s_j.pth is the pretrained model of trial i split j. Once 100 trials have been run, these files will be deleted, keeping only the files with the best performance and renamed to t_i_s_j_best.pth. i in range from 0 to 99 and j in range from 0 to 4.
    + t_i_s_j_best.pth is the pretrained model with hyperparameters of trial i split j that has the best performance on the validation set during cross validation. These models will be evaluated directly on the test set during testing.
    + t_i_final.pth is a pretrained model with hyperparameters of trial i, which has achieved the best performance in cross validation, retrained from scratch on the training and validation data set and evaluated on the test set.
    + scaler_raw_cv.pkl is the scaler of time series during cross validation.
    + scaler_pixel_cv.pkl is the scaler of images during cross validation.
- "-m" is run mode, you can only choose between "optimizer" or "evaluate". "optimizer" will try to create a database file that stores trials in the directory in argument "-d". If database file is already exist, will try to run finish 100 trials, and then train the model again from scratch with the best hyperparameters. "optimizer" is also the default value of this argument. "evaluate" will use the pretrained models named  t_i_s_j_best.pth and t_i_final.pth in the folder assigned with argument "-d" to evaluate on the test data set. If pretrained models are not available or trials are less than 100, "optimizer" will be used.

To rerun the code it is necessary to run this command line in the terminal, where *method.sh* could be short.sh, logmel.sh, conwave, recurrence.sh, gramian.sh or markov.sh depends on the method of converting time series to images. _approach_ represents the approach, which can be im_CNN, ts_im_CNN or tcl_CNN. *method_cv.py* could be stft_cv.py, lms_cv.py, cwt_cv.py, rp_cv.py, gaf_cv.py or mtf_cv.py. The commands below will try to check in the your_directory_name directory to see if there is a database file to save the trails. If not, it will create a new database file. If this file is available, it will see if 100 trails have been done, if not, it will run for 100 trials and find the best hyperparameter.
```bash
cd path/to/bachelorarbeit/approach/
```
if you use slurm to test the code.
```bash
sbatch method.sh -d your_directory_name -m optimize_or_evaluate
```
or if you use your local machine.
```bash
python3 method_cv.py -d your_directory_name -m optimize_or_evaluate
```
Since the command above only runs for one trial, this command can be used in the terminal for all 100 trials and for the final test. Note that this command line only supports slurm usage, where *METHOD_bash.sh* can be STFT_bash.sh, LMS_bash.sh, CWT_bash.sh, RP_bash.sh, GAF_bash.sh or MTF_bash.sh depends on the method of converting time series to images.
```bash
cd path/to/bachelorarbeit/approach/
```
```bash
bash METHOD_bash.sh -d your_directory_name -m optimize_or_evaluate
```
For convenience of testers who want to run the code and compare with our results, can use arguments:
- "-d" represents the name of the directory where you want to save the pretrained models, the database file that stores trials and scalers. The default value of this argument and also the default directory we use in this work is "model_pretrained". This folder contains:
    + METHOD.db is a database file containing trials including hyperparameters tested during cross validation and their results. METHOD is the abbreviation for the time sereries-to-image methods used in this work, for example STFT.db.
    + trials_data.csv is same as METHOD.db but in .csv
    + t_i_s_j.pth is the pretrained model of trial i split j. Once 100 trials have been run, these files will be deleted, keeping only the files with the best performance and renamed to t_i_s_j_best.pth. i in range from 0 to 99 and j in range from 0 to 4.
    + t_i_s_j_best.pth is the pretrained model with hyperparameters of trial i split j that has the best performance on the validation set during cross validation. These models will be evaluated directly on the test set during testing.
    + t_i_final.pth is a pretrained model with hyperparameters of trial i, which has achieved the best performance in cross validation, retrained from scratch on the training and validation data set and evaluated on the test set.
    + scaler_raw_cv.pkl is the scaler of time series during cross validation.
    + scaler_pixel_cv.pkl is the scaler of images during cross validation.
- "-m" is run mode, you can only choose between "optimizer" or "evaluate". "optimizer" will try to create a database file that stores trials in the directory in argument "-d". If database file is already exist, will try to run finish 100 trials, and then train the model again from scratch with the best hyperparameters. "optimizer" is also the default value of this argument. "evaluate" will use the pretrained models named  t_i_s_j_best.pth and t_i_final.pth in the folder assigned with argument "-d" to evaluate on the test data set. If pretrained models are not available or trials are less than 100, "optimizer" will be used.

To rerun the code it is necessary to run this command line in the terminal, where *method.sh* could be short.sh, logmel.sh, conwave, recurrence.sh, gramian.sh or markov.sh depends on the method of converting time series to images. _approach_ represents the approach, which can be im_CNN, ts_im_CNN or tcl_CNN. *method_cv.py* could be stft_cv.py, lms_cv.py, cwt_cv.py, rp_cv.py, gaf_cv.py or mtf_cv.py. The commands below will try to check in the your_directory_name directory to see if there is a database file to save the trails. If not, it will create a new database file. If this file is available, it will see if 100 trails have been done, if not, it will run for 100 trials and find the best hyperparameter.
```bash
cd path/to/bachelorarbeit/approach/
```
If you use slurm to test the code, the output will be saved in directory result/method
```bash
sbatch method.sh -d your_directory_name -m optimize_or_evaluate
```
Or if you use your local machine.
```bash
python3 method_cv.py -d your_directory_name -m optimize_or_evaluate
```
Or if you want to output as a text file, you can use the command.
```bash
python3 method_cv.py -d your_directory_name -m optimize_or_evaluate > out.txt
```
Since the command above only runs for one trial, this command can be used in the terminal for all 100 trials and for the final test. Note that this command line only supports slurm usage, where *METHOD_bash.sh* can be STFT_bash.sh, LMS_bash.sh, CWT_bash.sh, RP_bash.sh, GAF_bash.sh or MTF_bash.sh depends on the method of converting time series to images.
```bash
cd path/to/bachelorarbeit/approach/
```
```bash
bash METHOD_bash.sh -d your_directory_name -m optimize_or_evaluate
```
### pann
In this folder there will be the following files with the functions listed:
- pann.py: The time series will be converted to images using the Log Mel Spectrogram in model. This file includes cross validation with 100 trials, training on the training data set and testing on the test data set.
- pnn.sh: Job script to Slurm to run pann.py. Each run represents a trial in the hyperparameter tunning process.
- PANN.sh: Bash shell script to run multiple pnn.sh one after another. 

Same as above, to rerun the code, you need to enter the following command line into the terminal, two arguments -d and -m have the same function as mentioned above.
```bash
cd path/to/bachelorarbeit/pann/
```
If you use slurm to test the code, the output will be saved in directory result/pann
```bash
sbatch pnn.sh -d your_directory_name -m optimize_or_evaluate
```
Or if you use your local machine.
```bash
python3 pann.py -d your_directory_name -m optimize_or_evaluate
```
Or if you want to output as a text file, you can use the command.
```bash
python3 pann.py -d your_directory_name -m optimize_or_evaluate > out.txt
```
Since the command above only runs for one trial, this command can be used in the terminal for all 100 trials and for the final test. Note that this command line only supports slurm usage.
```bash
cd path/to/bachelorarbeit/pann/
```
```bash
bash PANN.sh -d your_directory_name -m optimize_or_evaluate
```
## Experiment tracking
If you want to have a closer look at each trial in each method as well as charts of metrics, you can access this project's experiment tracking through [neptune](https://ui.neptune.ai/auth/realms/neptune/protocol/openid-connect/auth?client_id=neptune-frontend&redirect_uri=https%3A%2F%2Fapp.neptune.ai&state=82f29ed0-7cf1-4cc1-b98e-6f3b9e664623&response_mode=fragment&response_type=code&scope=openid&nonce=bf90715e-8faf-4544-8f0c-5193e1388d9d) by logging in to account:

Username: anh-khoa

Password: khoa9898

Select workspace ba_final in the upper left corner of the screen. Projects named stft-1, lms-1, cwt-1, rp-1, gaf-1, mtf-1 belong to the im_CNN approach, while projects stft-2, lms-2, cwt-2, rp- 2, gaf-2, mtf-2 are ts_im_CNN approach and projects stft-3, lms-3, cwt-3, rp-3, gaf-3, mtf-3 come from tcl_im_CNN approach. Finally pann-2 belongs to the pann approach. The best_results directory contains visualizations of the best results of each approach.

## Conclusion
The table below shows the processing time from time series-to-image. In this work, we consider that STFT is the method with the best performance.
+--------+-----------------+-----------------+
| Method | Time Processing | Best score test |
+--------+-----------------+-----------------+
|  STFT  |      1h 15 m    |       0.37      |
+--------+-----------------+-----------------+
|   LMS  |     1 h 41 m    |       0.32      |
+--------+-----------------+-----------------+
|   CWT  |     7 h 25 m    |       0.34      |
+--------+-----------------+-----------------+
|   RP   |       45 s      |       0.34      |
+--------+-----------------+-----------------+
|   GAF  |       59 s      |       0.29      |
+--------+-----------------+-----------------+
|   MTF  |       63 s      |       0.34      |
+--------+-----------------+-----------------+