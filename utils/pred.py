import numpy as np
import os, csv, re, datetime
import sys, getopt
import h5py
from h5shape import *
from keras import models
from keras.models import load_model

#prediction times in test sets)
utcPlus2 = [30, 69, 126, 186, 234]
utcPlus3 = [57, 114, 174, 222, 258]

def load_test_data(file_path, indices):
    """Load data for one test day, return as numpy array with normalized samples.
    
        Args.:
            file_path (str): file path of h5 file for one day
            indices (list): list with prediction times (as list indices in the interval [0, 288])
            
        Returns: numpy array of shape (5, 3, 3, 495, 436)
    """
    #load h5 file
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])
    
    #identify test cases and split in samples of each length 3 time bins
    data = [data[y - 3: y] for y in indices]
    data = np.stack(data, axis=0)
    
    #transpose to (samples, timesteps, channels, rows, columns)
    data = np.transpose(data, (0, 1, 4, 2, 3))
    
    #rescale and return data
    data = data.astype(np.float32)
    np.random.shuffle(data)
    data /= 255.
    return data

def list_filenames(directory):
    filenames = os.listdir(directory)
    return filenames

def return_date(file_name):
    """Auxilliary function which returns datetime object from Traffic4Cast filename.
    
        Args.:
            file_name (str): file name, e.g., '20180516_100m_bins.h5'
        
        Returns: date string, e.g., '2018-05-16'
    """
    
    match = re.search(r'\d{4}\d{2}\d{2}', file_name)
    date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
    return date

def write_data(data, filename):
    """
    write data in gzipped h5 format.
    """
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

def write_output_files(model_dir, data_dir, output_dir, city):
    """
    write outdata into each submission folder structure at out_path, cloning
    filenames in corresponding folder structure at input_path.
    """
    
    test_indices = utcPlus3
    if city == "Berlin":
        test_indices = utcPlus2

    model_path = sys.argv[-1]
    model = load_model(model_path)

    # set relevant list
    test_data_dir = os.path.join(data_dir, city, city+"_test/")
    file_names = list_filenames(test_data_dir)
    for f in file_names:
        # load data
        x = load_test_data(test_data_dir + f, test_indices)
        out = model.predict(x)
        out *= 255
        out = out.astype(int)
        neg_indices = out < 0
        out[neg_indices] = 0
        out = np.transpose(out, (0, 1, 3, 4, 2))
        
        # output result
        outfile = os.path.join(output_dir, city, city+'_test',f)
        write_data(out, outfile)
        print("just wrote file {}".format(outfile))

        
# run prediction

if __name__ == '__main__':

    # gather command line arguments.
    data_dir = ''
    model_dir = ''
    output_dir = ''
    city = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:m:o:c:", ["data_dir","model_dir","output_dir","city"])
    except getopt.GetoptError:
        print('usage: pred.py -d <data dir> -m <model dir> -o <output dir> -c <city>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: pred.py -d <data dir> -m <model dir> -o <output dir> -c <city>')
        elif opt in ("-d","--data_dir"):
            data_dir = arg
        elif opt in ("-m","--model_dir"):
            model_dir = arg
        elif opt in ("-o","--output"):
            output_dir = arg
        elif opt in ("-c","--city"):
            city = arg
    if city in ("Berlin","Istanbul","Moscow"):
        write_output_files(model_dir, data_dir, output_dir, city)
    else:
        print('invalid city provided')
        sys.exit(2)
