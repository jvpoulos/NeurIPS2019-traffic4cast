import numpy as np
import os, csv, re, datetime
import sys, getopt
import h5py
from keras import models
from keras.models import load_model

#set indices (correspond to prediction times in test sets)
utcPlus2 = [30, 69, 126, 186, 234]
utcPlus3 = [57, 114, 174, 222, 258]

def load_data(file_path, indices):
    """Load data for one test day, return as numpy array with normalized samples of each
        6 time steps in random order.
    
        Args.:
            file_path (str): file path of h5 file for one day
            indices (list): list with prediction times (as list indices in the interval [0, 288])
            
        Returns: numpy array of shape (5, 6, 3, 495, 436)
    """
    #load h5 file
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])
    
    #identify test cases and split in samples of each length 3 time bins
    data = [data[y - 3 : y + 3] for y in indices]
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

def valid_loss_calc(data_dir, model_path, city):
    """
    write outdata into each submission folder structure at out_path, cloning
    filenames in corresponding folder structure at input_path.
    """
    
    test_indices = utcPlus3
    if city == "Berlin":
        test_indices = utcPlus2

    model = load_model(model_path)

    # set relevant list
    valid_data_dir = os.path.join(data_dir, city, city+"_validation/")
    file_names = list_filenames(valid_data_dir)
    
    city_mse = 0.0
    for f in file_names:
        # load data
        data = load_data(valid_data_dir + f, test_indices)

        #define first 3 time bins as input, last 3 as y_true
        x = data[:, :3]
        y_true = data[:, 3:]
        out = model.predict(x)

        # calculate MSE
        mse = (np.square(np.subtract(out,y_true))).mean(axis=None)
        city_mse += mse
    city_mse /= len(file_names)
    
    return(city_mse)

# run prediction

if __name__ == '__main__':

    # gather command line arguments.
    data_dir = ''
    model_dir = ''
    output_loss_path = ''
    city = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:m:o:c:", ["data_dir","model_dir","output_loss_path","city"])
    except getopt.GetoptError:
        print('usage: pred.py -d <data dir> -m <model dir> -o <output loss path> -c <city>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: pred.py -d <data dir> -m <model dir> -o <output loss path> -c <city>')
        elif opt in ("-d","--data_dir"):
            data_dir = arg
        elif opt in ("-m","--model_dir"):
            model_dir = arg
        elif opt in ("-o","--output_loss_path"):
            output_dir = arg
        elif opt in ("-c","--city"):
            city = arg
    if city in ("Berlin","Istanbul","Moscow"):
        #generate loss file
        loss_file = open(output_loss_path + city + "_valid_loss.txt", 'w', newline='' )
        loss_writer = csv.writer(loss_file, lineterminator="\n")
        loss_writer.writerow(['epoch','loss'])

        for epoch in range(0, 26):
            print(epoch)
            model_path = model_dir + "Istanbul/model_ep_" + str(epoch) + '.h5'
            city_mse = valid_loss_calc(data_dir, model_path, city)
            loss_writer.writerow([epoch, city_mse])
    
        loss_file.close()
    else:
        print('invalid city provided')
        sys.exit(2)