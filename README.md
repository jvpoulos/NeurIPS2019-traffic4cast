# Traffic4cast benchmarks and submission utilities for core competition.

## Introduction

The aim of our core competition is to predict the next 3 images (5 times on a given test day) 
in our traffic movies, which encode the volume, speed and direction of observed traffic in each 100m x 100m grid 
within a 5min interval into an RGB pixel colour, as described, for Berlin, Istanbul and Moscow.

The submission format for the 5 sequences of 3 images in each day of the test set is a multi-dimensional
array (tensor) of shape (5,3,495,436,3) and the objective function of all submitted tensors (one for each day
in the test set list and each city) is the mean squared error of all pixel channel colour values to pixel colour
values derived from true observations. We note that we normalize these pixel colour values to lie between 0 and 1
by dividing the pixel colour value (between 0 and 255) by 255.

## Benchmarks

The attached code provides the following benchmarks.

### Submission of only zeros.
Given that many pixel values are (0,0,0) (mostly due to absence of drivable road network in the underlying 100mx100m area) just submitting
all zero values will result in many of the terms in the MSE calculation to be zero. Use the create_submissiontest_like.py script in utils
to generate the necessary submission file as described, via
```
python3 create_submissiontest_like -i <path to competition data> -o ./test_submission_data_with_constant_zero -v 0
```
which creats the output directory "test_submission_data_with_constant_zero" in the current directory. Then produce a submission file, for instance
via
```
cd test_submission_data_with_constant_zero; zip -r ../constant_zero_submission.zip .; cd ..
```
which creates the submission file "constant_zero_submission.zip" (in the original current folder). You can submit this file to the competition as a test via
[here](https://www.iarai.ac.at/traffic4cast/competitions/traffic4cast-2019-core/)

### Simple average of 3 previous images.
Our second benchmark is the prediction obtained by averaging the 3 previous image pixel values in all colour channels to obtain an estimate
of the next image's corresponding value. Use the script "naive_baseline_mavg.py" in the benchmark/naive_moving_average subfolder to generate 
an output folder and then zip that folder as show below into a submission file.

### Seq2Seq network taking 3 input images and predicting the next 3 trained on entire training data.
The file "baseline_seq2seq.py" contains a sample training code for 2 layered ConvLSTM Seq2Seq model which is trained on two GPUs. The syntax to run train this model is
```
python3 baseline_seq2seq.py -d <path to competition data> -m <directory in which model checkpoints are saved> -l <path to log file> - c <city to be trained>
```
We will provide trained parameters and a submission file generation script in due course.

### Seq2Seq network taking 3 input images and predicting the next 3 trained only on the corresponding time slots for the submission.
The script "baseline_seq2seq_foc_train.py" is nearly the same script as "baseline_seq2seq.py", execpt it restricts training only to those time windows to be predicted 
in our competition. It has the same syntax as the script above and a submission file generator is provided in due course.  


## Submission guide.

Currently, the competition data provided comes in a zip file that has the following folder structure.
```
+-- Berlin +-- Berlin_training  -- ...
	   +-- Berlin_validation -- ...
	   +-- Berlin_test-- + -- 20180102_100m_bins.h5
			     + -- 20180108_100m_bins.h5
					...
			     + -- 20181223_100m_bins.h5

+-- Istanbul +-- Istanbul_training  -- ...
	     +-- Istanbul_validation -- ...
	     +-- Istanbul_test-- + -- 20180104_100m_bins.h5
			         + -- 20180107_100m_bins.h5
					...
		                 + -- 20181223_100m_bins.h5
	
  
+-- Moscow +-- Moscow_training  -- ...
	   +-- Moscow_validation -- ...
	   +-- Moscow_test-- + -- 20180106_100m_bins.h5
			     + -- 20180110_100m_bins.h5
					...
		             + -- 20181227_100m_bins.h5
```
and each of the files 2018mmdd_100m_bins.h5 is a h5 encoding of an int16 tensor of shape (288, 495, 436, 3) corresponding to the 288 RBG images that
make up the 5 minute intervals of the 495 x 436 colour images (in order) with respect to a UTC time stamp. Thus the tensor [0,:,:,:] contains the RGB
image encoding the traffic conditions from time interval 0:00 UTC to 0:05 UTC of the corresponding city on the data corresponding to the file name.
For the submission, we expect a zip file back that, when unpacked, decomposes into the following folder structure:
```
+-- Berlin +-- Berlin_test-- + -- 20180102_100m_bins.h5
			     + -- 20180108_100m_bins.h5
					...
			     + -- 20181223_100m_bins.h5

+-- Istanbul +-- Istanbul_test-- + -- 20180104_100m_bins.h5
			         + -- 20180107_100m_bins.h5
					...
		                 + -- 20181223_100m_bins.h5
	
  
+-- Moscow +-- Moscow_test-- + -- 20180106_100m_bins.h5
			     + -- 20180110_100m_bins.h5
					...
		             + -- 20181227_100m_bins.h5
```
where now each file 2018mmdd_100bins.h5 contains a uint8 tensor of shape (5, 3, 495, 436, 3) that contains the 5 predictions of 3 successive images
following the 5 sequences of 12 non-zero images given in the files of the Berlin/Berlin_test, Istanbul/Istanbul_test and Moscow/Moscow_test directories
of the competition data.
The submission file can uploaded [here](https://www.iarai.ac.at/traffic4cast/competitions/traffic4cast-2019-core/).


### create_submissiontest_like.py -i input_directory -o output_directory -v value or "random"
This script clones the submission folder structure and file names it finds in the subfolders Berlin/Berlin_test, Istanbul/Istanbul_test and Moscow/Moscow_test of the input_folder 
to the output folder, where the corresponding files are h5 files each containing a uint8 (5, 3, 495, 436, 3) tensor either containing the constant value given in the -v option, or,
if the latter contained the word "random", uniformly randomly chosen uint8s between 0 and 255 for each entry. To generate a submission zip file, cd into the output folder
and zip the 3 folders and their subfolders into an output file, e.g. using the command
```
zip -r ../constant_zero_submission.zip .
```
will produce the submission file "constant_zero_submission.zip" in the same directory as the output folder.

### h5shape.py -i input_file
This script expects an h5 file and will print out the tensor shape. For the test files in the competition data folder, for instance, 
```
python3 h5file.py -i path_to_competition_files/Berlin/Berlin_test/20180102_100m_bins.h5
```
will produce the output 
```
(1, 288, 495, 436, 3)
```
The output for the corresponding submission file would be (1, 5, 3, 495, 436, 3).
from a linux command line.

### eval.py -g path_to_eval_folder1 -s path_to_eval_folder2 -o output_folder -p prefix
This script assumes that the folders provided via the -g (golden) and -s (submitted) options contain the submission file format structure and compares
these two - file by file, city by city and overall - by calculating the corresponding MSE. The results are output in 3 files with file name prefix.score, prefix.extended_score
and prefix.log that contain the overall MSE, the overall MSE per city and the individual MSEs, respectively.

### valid_loss.py -d data_folder -m fitted_model_dir -o output_folder -c city
This script calculates the MSE of trained models on the validation datasets

### pred.py -d data_folder -m fitted_model_dir -o output_folder -c city model_path
This script generates prediction for submission based on the trained model

## License

Apache 2.0
