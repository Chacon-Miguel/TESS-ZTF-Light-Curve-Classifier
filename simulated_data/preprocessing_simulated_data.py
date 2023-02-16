"""
See if you can make that in the same format to pass through network

Pick a subset to pass in everything
goal: get numpy array of the following dimensions (# of light_curves, max light curve length, measurements which are arrays of 4 numbers)

get light curves to be between (-30, 70)

"""
import numpy as np
import os
import pickle as p
import astropy
from matplotlib import pyplot as plt
import random

def get_lc_lengths():
    total = 0
    lengths = {}
    # every index is a bin of 50, so for index 1, it will hold number of light curves with length
    # at most 50, for index 2, it will hold number of light curves with length at most 100
    bins = [0]*201
    file_directory = "C:/Users/Miguel A Chacon/Downloads/light_curves_sims/"
    for pfile in os.listdir(file_directory):
        if '.pickle' in pfile:
            print(pfile)
            with open(file_directory + pfile, 'rb') as file:
                data = p.load(file)
                # get length of every light curve 
                # when restricted to -30 < t < 70
                for (name, table) in data.items():
                    length = len([data_point for data_point in table['time'] if -30 < data_point < 70])
                    lengths[length] = lengths.get(length, 0) + 1
                total += len(data)
    # now sort the lengths dictionary
    lengths_sorted = sorted(lengths.items(), key= lambda data_point: data_point[0])
    # separate them to plot later
    lengths = []
    amts = []
    # now put the lengths into bins
    last_bin = 0
    curr_bin = 1
    # iterate through sorted lengths and add to bins
    for (length, amt) in lengths_sorted:
        lengths.append(length)
        amts.append(amt)
        if length <= curr_bin*10:
            bins[curr_bin] += amt
        else:
            bins[curr_bin] += bins[last_bin]
            last_bin += 1
            curr_bin += 1
    for i in range(1, len(bins)):
        if bins[i] == 0:
            bins[i] = bins[i-1]
    # get the fraction that the bin gets of the total dataset
    bins = [bin/total*100 for bin in bins]
    # now plot them
    # ax1 will be lengths and ax2 will be bins
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(lengths, amts, s=3)
    ax2.scatter(list(range(0, 2010, 10)), bins)
    fig.show()
    fig.savefig(file_directory + 'lengths_and_bins_plots.png')
    bins = [bin*total/100 for bin in bins]
    print(bins)


def process_data(final_length):
    """
    Goes through every pickle file in save light curves newsims folder. Every file holds a
    dictionary with the light curve name to an astropy table that holds all the data of the light curve
    
    Only the first four columns are needed and rearranged in the way the neural networks expects the input to be.
    
    Every light curve that is at most 300 datapoints is converted into a numpy array

    I need one dictionary that's from classification to light curves
    I also need the light curve names
    and a numpy array of light curve classifications


    """
    # result = {}
    numerical_encodings = {
        'g': 4.8,
        'r': 6.5
    }
    class_name_map={
        '90':      'SNIa-norm', 
        '67':      'SNIa-91bg', 
        '52':      'SNIa-x', 
        '60':      'SLSN-I', 
        '64':      'TDE', 
        '99':      'Kilonova', 
        'II':    'II', 
        'IIn':   'II', 
        'IIb':   'Ibc', 
        'Ib':    'Ibc', 
        'Ic':    'Ibc', 
        'Ic-BL': 'Ibc'
    }
    SNIa = {'SNIa', 'SNI', 'SNIa-91T-like',
            'SNIa-91bg-like', 'SNIa-pec', 'SNIa-SC', 'SNIa-x', 'SNIa-91bg', 'SNIa-norm'}
    SNIbc = {'SNIbn', 'SNIb/c', 'SNIb', 'SNIc', 'SNIc-BL', 'Ibc', 'SNIbc'}
    SNIi = {'SNII', 'SNIIb', 'SNIIP', 'SNII-pec', 'SNIIn', 'II'}
    other = {'CV', 'SLSN-I', 'AGN', 'FRB', 'Mdwarf',
            'Nova', 'Other', 'Varstar', 'TDE', 'Kilonova', 'CCSN'}
    file_directory = "C:/Users/Miguel A Chacon/Downloads/light_curves_sims/"
    # first list will hold names of light curves
    # second list will hold classifications of light curves
    # third will be all light curves
    light_curves = [[], [], []]

    # for every pickle file...
    for pfile in os.listdir(file_directory):
        if '.pickle' in pfile:
            # will get the class number like 52 or 42 with the .pickle extension
            # so class number wil always be something like 52.pickle
            class_number = pfile.split('_')[-1]
            # remove the .pickle extension
            class_number = class_number[:class_number.find('.')]
            class_name = class_name_map[class_number]
            print(class_number)

            # data is dictionary of string -> astropy table
            # i.e name of object to light curve
            file = open(file_directory + pfile, 'rb')
            lc_dict = p.load(file)
            file.close()
            result = {}
            # for every light curve...
            for (name, table) in lc_dict.items():
                light_curve = []
                for [band, time, flux, error, photFlag] in table:
                    if -30 < time < 70:
                        light_curve.append([time, numerical_encodings[band], flux, error]) 
                    # if time >= 70:
                    #     break
                if len(light_curve) <= final_length and len(light_curve) > 40:
                    # add padding if needed
                    while len(light_curve) < final_length:
                        light_curve.append([0.0, 0.0, 0.0, 0.0])
                    # light_curve = np.array(light_curve)

                    result[name] = light_curve
                    light_curves[0].append(name)
                    light_curves[1].append(class_name)
                    light_curves[2].append(light_curve)
                    
            with open(file_directory+'separated_classes/numpy_files/lc_class_name_' + class_name + ".pickle", 'wb') as file:
                p.dump(result, file)

    # concatenante all light curves into one numpy array
    light_curves[0] = np.array(light_curves[0])
    light_curves[1] = np.array(light_curves[1])
    light_curves[2] = np.array(light_curves[2])
    print([array.shape for array in light_curves])
    with open(file_directory + 'separated_classes/numpy_files/entire_dataset.pickle', 'wb') as file:
        p.dump(light_curves, file)


def get_random_sets(samples):
    """
    Takes in the whole dataset and returns random light curves of each of the 4 categories
    """
    # define categories
    SNIa = {'SNIa', 'SNI', 'SNIa-91T-like',
            'SNIa-91bg-like', 'SNIa-pec', 'SNIa-SC', 'SNIa-x', 'SNIa-91bg', 'SNIa-norm'}
    SNIbc = {'SNIbn', 'SNIb/c', 'SNIb', 'SNIc', 'SNIc-BL', 'Ibc'}
    SNIi = {'SNII', 'SNIIb', 'SNIIP', 'SNII-pec', 'SNIIn', 'II'}
    other = {'CV', 'SLSN-I', 'AGN', 'FRB', 'Mdwarf',
            'Nova', 'Other', 'Varstar', 'TDE', 'Kilonova'}
    class_encodings = ['II', 'Ibc', 'Kilonova', 'SLSN-I', 'SNIa-91bg', 'SNIa-norm', 'SNIa-x', 'TDE']
    categories = {class_name:[] for class_name in class_encodings}
    def modify_class(name):
        """
        Numerically encodes class names
        """
        if name in SNIa:
            return 0
        elif name in SNIbc:
            return 1
        elif name in SNIi:
            return 2
        elif name == 'Unclassified':
            return 3
        else:
            return 4
    file_directory = "C:/Users/Miguel A Chacon/Downloads/light_curves_sims/separated_classes/numpy_files/"
    with open(file_directory + 'entire_dataset.pickle', 'rb') as file:
        light_curves = p.load(file)

        for i in range(len(light_curves[0])):
            name, class_name, light_curve = light_curves[0][i], light_curves[1][i], light_curves[2][i]
            categories[class_name].append(i)
            # light_curves[1][i] = modify_class(class_name)
    # first one will hold names
    # second one will hold labels
    # third one will hold light curves
    for category, data_points in categories.items():
        print(category, len(data_points))
    dataset = [[], [], []]

    # get 2000 random points for each of the 4 categories
    for category, data_points in categories.items():
        if category == 3:
            continue
        chosen_dps = random.sample(data_points, k = samples)
        for i in chosen_dps:
            for j in range(3):
                dataset[j].append(light_curves[j][i])
    # concatenate all numpy arrays
    dataset[0] = np.array(dataset[0])
    dataset[1] = np.array(dataset[1])
    dataset[2] = np.array(dataset[2])
    # dataset len should be 12K
    print(len(dataset[0]))
    
    with open(file_directory + 'dataset.pickle', 'wb') as file:
        p.dump(dataset, file)

# def plot_light_curves():

# get_lc_lengths()

# process_data(80)

get_random_sets(2000)
