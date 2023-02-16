import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

"""
tns_info.csv holds the labels and names of all light curves
And then there's a folder with all of the light curve data
"""

file_path = "C:/Users/Miguel A Chacon/Downloads/"

def get_labels():
    # create dataframe of table that holds name and classification
    lc_info = pd.read_csv(file_path + 'tns_info.csv', usecols= ['IAU_name','classification'])
    name_to_label_map = {}
    # count number of labeled and unlabeled light curves
    unlabeled_counter = 0
    labeled_counter = 0
    for index, data in lc_info.iterrows():
        name, labels = data
        print(name)
        # check if light curve is classified
        if pd.isna(labels):
            name_to_label_map[name] = 'unclassified'
            unlabeled_counter += 1
        else:
            # if classified, they're could be multiple classifications
            # that are usually the same
            # in which case, just take the first label (that's the labels.split('/')[1]) 
            # example of multiple labels: /SN II/Other
            label = labels.split('/')[1].replace(" ", '')
            name_to_label_map[name] = label
            labeled_counter += 1

    print('Labeled light curves:', labeled_counter)
    print('Unlabeled light curves:', unlabeled_counter)

    return name_to_label_map

# name_to_label_map = get_labels()

def plot_lengths(bin_size, max_length):
    """"
    Now get light curve data from raw_light_curve_data
    """

    # lengths
    total = 0
    lengths = {}
    # every index is a bin of 50, so for index 1, it will hold number of light curves with length
    # at most 50, for index 2, it will hold number of light curves with length at most 100
    print('bins list size', ((max_length+bin_size)//bin_size))
    bins = [0]*((max_length+bin_size)//bin_size)
    band_to_number_map = {'tess':7.9, 'g-band':4.8, 'r-band':6.5}

    for file_name in os.listdir(file_path + 'raw_light_curve_data'):
        lc_name = file_name.split("_")[1]
        lc_dframe = pd.read_csv(file_path + 'raw_light_curve_data/' + file_name, usecols=['relative_time', 'tess_flux', 'r_flux', 'g_flux', 'tess_uncert', 'r_uncert', 'g_uncert'])
        lc_arr = []

        for index, data in lc_dframe.iterrows():
            time, tess_flux, r_flux, g_flux, tess_err, r_err, g_err = data
            # restrict to times between [-30, 70]
            if -30 > time or time > 70:
                continue

            if not pd.isna(tess_flux):
                lc_arr.append( [time, band_to_number_map['tess'], tess_flux, tess_err] )
            
            if not pd.isna(g_flux):
                lc_arr.append( [time, band_to_number_map['g-band'], g_flux, g_err] )
            
            if not pd.isna(r_flux):
                lc_arr.append( [time, band_to_number_map['r-band'], r_flux, r_err] )
        # update lengths dictionary
        length = len(lc_arr)
        lengths[length] = lengths.get(length, 0) + 1
        total += 1
    print("Total Light Curves:", total)
    # now sort the lengths dictionary
    lengths_sorted = sorted(lengths.items(), key= lambda data_point: data_point[0])

    # now put the lengths into bins
    j = 0
    # iterate through sorted lengths and add to bins
    for curr_bin, bin in enumerate(bins):
        while j < len(lengths_sorted):
            length, amt = lengths_sorted[j]
            print(curr_bin*bin_size, length, amt, bins[curr_bin])
            if length <= curr_bin*bin_size:
                bins[curr_bin] += amt
            else:   
                break
            j += 1
        bins[curr_bin] += bins[curr_bin-1]
    
    # get the fraction that the bin gets of the total dataset
    bins_normalized = [bin/total*100 for bin in bins]
    # now plot them
    # ax1 will be lengths and ax2 will be bins
    fig, (ax2, ax3) = plt.subplots(2)
    # ax1.scatter(list(lengths.keys()), list(lengths.values()), s=3)
    ax2.scatter(list(range(0, max_length+bin_size, bin_size)), bins_normalized)
    ax3.scatter(list(range(0, max_length+bin_size, bin_size)), bins)
    fig.show()
    fig.savefig(file_path + 'lengths_and_bins_plots_real_data.png')
    # bins = [bin*total/100 for bin in bins]
    # print(bins)
    # save to text file
    bin_sizes = list(range(0, max_length+bin_size, bin_size))
    with open(file_path + "bins_plot_datapoints.txt", 'w') as file:
        file.write('Bin Size    Light Curves Captured\n')
        for i, bin_val in enumerate(bins):
            file.write(" %d:\t\t%d\n" % (bin_sizes[i], bin_val))
    with open(file_path + "bins_percentage_plot_datapoints.txt", 'w') as file:
        file.write('Bin Size    Percent of Dataset Captured\n')
        for i, bin_val in enumerate(bins_normalized):
            file.write(" %d:\t\t%d\n" % (bin_sizes[i], bin_val))

    # np.savetxt(file_path + 'bins_percentage_plot_datapoints.txt', np.array(bins_normalized), delimiter= " ")
    # np.savetxt(file_path + 'bins_plot_datapoints.txt', np.array(bins), delimiter= " ")

bin_size = 10
max_length = 300
plot_lengths(bin_size, max_length)