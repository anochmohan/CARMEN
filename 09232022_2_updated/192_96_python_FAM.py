#!/usr/bin/env python
#
# Copied largely from the Jupyter notebook 'Copy of CFL048_Biomark_csv_v3.ipynb' in a
# Google Drive that Nicole W sent
#
# Run like:
#  ```parse_data.py --fcount ADAPTF001 --chip-dim 96 --in-csv data/20200728_1362564078_rawdata.csv --in-layout data/ADAPTF001_96_assignment.xlsx --timepoints 25 --out-dir output/```

import pandas as pd
import numpy as np
from os import listdir,path
import warnings
import math
import csv
import argparse
import gzip
from sys import exit
import plotly.graph_objects as go

# Read args
parser = argparse.ArgumentParser()
parser.add_argument('--fcount',
                    required=True,
                    help="Experiment count (e.g., 'CFL091')")
parser.add_argument('--chip-dim',
                    required=True,
                    choices=['96', '192'],
                    help=("Fluidigm chip type (number of samples/targets)"))
parser.add_argument('--timepoints',
                    type=int,
                    default=25,
                    help="Number of timepoints")
parser.add_argument('--in-csv',
                    required=True,
                    help=("CSV file with data"))
parser.add_argument('--in-layout',
                    required=True,
                    help=("Excel sheet giving layout"))
parser.add_argument('--out-dir',
                    required=True,
                    help=("Output directory"))
args = parser.parse_args()

try:
    args = parser.parse_args()
except AttributeError:
    parser.print_help()
    exit()
#if not args.out_dir:
#    parser.print_help()
#    exit()

# Set up
ifc = args.chip_dim # 96 or 192
instrument_type = 'BM' # EP1 or BM (Biomark)
fcount = args.fcount # experiment file that will be used
tgap = 1 # time gap between mixing of reagents (end of chip loading) and t0 image in minutes

if instrument_type == 'EP1':
    count_tp = 1 # End point run
else:
    count_tp = args.timepoints # number of timepoints, standard for 2h techdev is 25 atm

# Define variables based on inputs
exp_name = fcount + '_' + ifc
#csv_file = fcount + '_' + ifc + '_' + instrument_type +'_rawdata.csv' # csv file with raw data from Fluidigm RT PCR software
csv_file = args.in_csv

# Write a path to that excel sheet
layout_file = args.in_layout # uses 192_assignment.xlsx for layout

out_folder = args.out_dir # actual directory is called output!!!!!!!!!!!!!!!!!!

if path.exists(layout_file) == False:
    raise Exception((layout_file + " doesn't exist")) # checks for 192_assignement.xlsx file and raise exception when false
    
if path.exists(csv_file) == False:
    raise Exception((csv_file + " doesn't exist")) # checks for the data sheet from instrument. In this case the file is 1691383040.csv

# Definition of functions
# Create a dictonary for timepoints, creates a key/value pair (t1/4), (t2/9), (t3/14) etc
time_assign = {}
for cycle in range(1,38):
    tpoint = "t" + str(cycle)
    time_assign[tpoint] = tgap + 3 + (cycle-1) * 5
# calculate the real timing of image
# used for image and axis labeling 
def gettime(tname):
    realt = time_assign[tname] # a key is used on the dictonary to get the value, which is stored in realt 
    return (realt)             # returns the value

if instrument_type == 'BM':
    if ifc == '96':
        probe_df = pd.read_csv(csv_file,header = 18449, nrows = 9216) # FAM
        reference_df = pd.read_csv(csv_file, header = 9231, nrows = 9216) # ROX
        bkgd_ref_df = pd.read_csv(csv_file, header = 27667, nrows = 9216)
        bkgd_probe_df = pd.read_csv(csv_file,header = 36885, nrows = 9216)
    if ifc == '192':
        #probe_df = pd.read_csv(csv_file,header = 9233, nrows = 4608)
        #reference_df = pd.read_csv(csv_file, header = 4623, nrows = 4608)
        #bkgd_ref_df = pd.read_csv(csv_file, header = 13843, nrows = 4608)
        #bkgd_probe_df = pd.read_csv(csv_file,header = 18453, nrows = 4608)
        probe_df = pd.read_csv(csv_file, sep=",", header = 9238, nrows = 4608, skip_blank_lines=False)
        reference_df = pd.read_csv(csv_file, sep=",", header = 4627, nrows = 4608, skip_blank_lines=False)
        bkgd_ref_df = pd.read_csv(csv_file, sep=",", header = 13849, nrows = 4608, skip_blank_lines=False)
        bkgd_probe_df = pd.read_csv(csv_file, sep=",", header = 18460, nrows = 4608, skip_blank_lines=False)
elif instrument_type == 'EP1':
    if ifc == '192':
        probe_df = pd.read_csv(csv_file,header = 9238, nrows = 4608)
        reference_df = pd.read_csv(csv_file, header = 4628, nrows = 4608)
        bkgd_ref_df = pd.read_csv(csv_file, header = 18458, nrows = 4608)
        bkgd_probe_df = pd.read_csv(csv_file,header = 23068, nrows = 4608)
        
# Get rid of stuff
c_to_drop = 'Unnamed: ' + str(count_tp+1)
probe_df = probe_df.set_index("Chamber ID").drop(c_to_drop, axis=1) # Chamber ID is set as the first column using .set_index(). .drop returns series with specified index label removed. So it removes columns labelled “unnamed: 1” etc. 
reference_df = reference_df.set_index("Chamber ID").drop(c_to_drop, axis=1)
bkgd_ref_df = bkgd_ref_df.set_index("Chamber ID").drop(c_to_drop, axis=1)
bkgd_probe_df = bkgd_probe_df.set_index("Chamber ID").drop(c_to_drop, axis=1)

probe_df.columns = probe_df.columns.str.lstrip() # remove spaces from beginning of column names
reference_df.columns = reference_df.columns.str.lstrip() 
bkgd_ref_df.columns = bkgd_ref_df.columns.str.lstrip()
bkgd_probe_df.columns = bkgd_probe_df.columns.str.lstrip()

# rename column names
probe_df.columns = ['t' + str(col) for col in probe_df.columns]
reference_df.columns = ['t' + str(col) for col in reference_df.columns]
bkgd_ref_df.columns = ['t' + str(col) for col in bkgd_ref_df.columns]
bkgd_probe_df.columns = ['t' + str(col) for col in bkgd_probe_df.columns]

# if an error like "Passed header=9233 but only 4618 lines in file" comes up, you probably exported the wrong csv file from the RT-PCR software (you need 'table results with raw data')
# if an error like ""['Unnamed: 26'] not found in axis" "comes up, look at the probe_df and look at the last column. Likely, the number of timepoints is wrong

# Substract the background from the probe and reference data
probe_bkgd_substracted = probe_df.subtract(bkgd_probe_df) #probe_df – bkgd_probe_df (this is what is being done, to subtract arrays, we use the subtract func.)
ref_bkgd_substracted = reference_df.subtract(bkgd_ref_df)

# Normalize the probe signal with the reference dye signal
signal_df = pd.DataFrame(probe_bkgd_substracted/ref_bkgd_substracted) # a table is created with the normalized values

# reset index
signal_df = signal_df.reset_index()

# split Column ID into SampleID and AssayID
splitassignment = signal_df['Chamber ID'].str.split("-",n=1,expand=True) #Chamber ID is split into two at “-“  … split into two indices cuz of n=1. 
signal_df["sampleID"] = splitassignment[0]
signal_df["assayID"] = splitassignment[1]

#set index again to Chamber ID
signal_df = signal_df.set_index('Chamber ID')

sampleID_list = signal_df.sampleID.unique()
assayID_list = signal_df.assayID.unique()

# Save csv
signal_out_csv_1 = f"{out_folder}/{exp_name}_{instrument_type}_1_signal_bkgdsubtracted_norm_{str(count_tp)}.csv"
#signal_df.to_csv(path.join(out_folder, exp_name+ '_' +instrument_type + '_1_signal_bkgdsubtracted_norm_' + str(count_tp) +'.csv')) # .to_csv – creates new csv file with the table. path.join adds the csv file into the out_folder with the suggested name
signal_df.to_csv(signal_out_csv_1)
# Create two dictionaries that align the IFC wells to the sample and assay names
#samples_layout_wo_string = pd.read_excel(path.join('',layout_file),sheet_name='layout_samples')
samples_layout_wo_string = pd.read_excel(f"{layout_file}",sheet_name='layout_samples', engine='openpyxl')
samples_layout = samples_layout_wo_string.applymap(str)                                        #applymap method applies a function that accepts and returns a scalar to every element of a DataFrame.
#assays_layout_wo_string = pd.read_excel(path.join('',layout_file),sheet_name='layout_assays')
assays_layout_wo_string = pd.read_excel(f"{layout_file}",sheet_name='layout_assays', engine='openpyxl')
assays_layout = assays_layout_wo_string.applymap(str)

# Create a dictionary with assay numbers and their actual crRNA / target name
#assays = pd.read_excel(path.join('',layout_file),sheet_name='assays')
assays = pd.read_excel(f"{layout_file}",sheet_name='assays', engine='openpyxl')
assays_zip = zip(assays_layout.values.reshape(-1),assays.values.reshape(-1))
assays_dict = dict(assays_zip)

#samples = pd.read_excel(path.join('',layout_file),sheet_name='samples')
samples = pd.read_excel(f"{layout_file}",sheet_name='samples', engine='openpyxl')

#function for appending numbers to repeated NTC/blank samples
c = 1
def append_num_to_ntc(x):
    global c
    if x == 'NTC' or x == 'blank':
        x = f"{x}_{c}"
        c += 1
    return x

#updating samples using the function
samples = samples.applymap(lambda x: append_num_to_ntc(x))

# Create a dictionary with sample numbers and their actual sample name
samples_zip = zip(samples_layout.values.reshape(-1),samples.values.reshape(-1))
samples_dict = dict(samples_zip)

# Map assay and sample names
signal_df['assay'] = signal_df['assayID'].map(assays_dict)
signal_df['sample'] = signal_df['sampleID'].map(samples_dict)

# Save csv
signal_out_csv_2 = f"{out_folder}/ {exp_name}_{instrument_type}_2_signal_bkgdsubtracted_norm_named_{str(count_tp)}.csv"
#signal_df.to_csv(path.join(out_folder, exp_name+'_' +instrument_type +'_2_signal_bkgdsubtracted_norm_named_' + str(count_tp) +'.csv'))
signal_df.to_csv(signal_out_csv_2)

# # Transform and summarize data for plotting

#create list with timepoints
# count_tp = 23 # in case you were wrong before
t_names = []
for i in range(1,count_tp+1):
    t_names.append(('t' + str(i)))

# Create a list of all assays and samples
# only indicate columns with unique assays. np.unique could be used on the list, but messes up our prefered the order
if ifc == '96':
    new_array = np.stack(assays[['C1','C2','C3','C4','C5']].values,axis=-1)
if ifc == '192':
    new_array = np.stack(assays[['C1','C2', 'C3']].values,axis=-1)

assay_list = np.concatenate(new_array).tolist()
print('identified crRNAs: ',len(assay_list))

# Do this for the samples
if ifc == '96':
    new_array = np.stack(samples[['C1','C2','C3','C4','C5','C6','C7', 'C8','C9','C10','C11','C12']].values,axis=-1)
if ifc == '192':
    new_array = np.stack(samples[['C1','C2','C3','C4','C5','C6', 'C7','C8','C9','C10','C11','C12', 'C13','C14','C15','C16','C17','C18', 'C19','C20','C21','C22','C23','C24']].values,axis=-1)
#     new_array = np.stack(samples[['C1','C2','C3','C4','C5','C6',\
#                                   'C7','C8','C9','C10','C11','C12']].values,axis=-1)

sample_list = np.concatenate(new_array).tolist()
print('identified samples: ',len(sample_list))

# Grouped medians 
grouped_stuff = signal_df.groupby(['assay','sample'])
medians = grouped_stuff.median(numeric_only = True)

# Creating dataframes in a loop: 
# https://stackoverflow.com/questions/30635145/create-multiple-dataframes-in-loop
med_frames = {}
for name in t_names:
    #time_med = signal_df.groupby(['assay','sample']).median()[name].unstack()
    time_med = medians[name].unstack()
    time_med.index.names=['']
    time_med.columns.names=['']
    med_frames[name] = time_med


# Write results

# Write TSV, with one row per (timepoint, sample (target), assay (guide)) triplet
# The value written is: {median across replicates[(probe signal - probe background) / (reference signal - reference background)]}
# for different time points

#with gzip.open(path.join(out_folder, exp_name+ '_'+ instrument_type + '_merged.tsv.gz'), 'wt') as fw:
with gzip.open(f"{out_folder}/{exp_name}_{instrument_type}_merged.tsv.gz", 'wt') as file:    
    def write_row(row):
        file.write('\t'.join(str(x) for x in row) + '\n')
    header = ['timepoint', 'minute', 'guide', 'target', 'value']
    write_row(header)

    for tp in med_frames.keys():
        rt = gettime(tp)
        for target in sample_list:
            for guide in assay_list:
                v = med_frames[tp][target][guide]
                row = [tp, rt, guide, target, v]
                write_row(row)




# Plotly heatmap

# Set x-axis, y-axis, and plot values
fig = go.Figure(go.Heatmap(
    x=med_frames[tp][sample_list].reindex(assay_list).columns.tolist(),
    y=med_frames[tp][sample_list].reindex(assay_list).index.tolist(),
    z=med_frames[tp][sample_list].reindex(assay_list),
    colorbar=dict(y=0.5, len=1)))

# Update heatmap with labels and margins
fig.update_layout(
    yaxis=dict(tickmode='linear', tick0=1, dtick=1, autorange='reversed'),
    xaxis=dict(tickmode='linear'),
    xaxis_title='Samples',
    yaxis_title='Assays',
    title_text=f'{exp_name} {str(rt)}min - median values', title_x=0.95, title_y=0.8,
    autosize=False,
    width=2700,
    height=700,
    margin=dict(l=50, r=50, b=50, t=175, pad=5)
) 

# Add annotationd
details = {
    # Set the cordinate
    'x': 0.0,
    'y': 1.50,
    'xref': 'paper',
    'yref': 'paper',
    'xanchor': 'auto',
    'yanchor': 'top',

    # Set format for text and box
    'text': f'Layout File Read: {layout_file} <br> CSV File Read: {csv_file} <br> Output Files: {signal_out_csv_1}, {signal_out_csv_2}, and {exp_name}_{instrument_type}_merged.tsv.gz <br> Arguments: python 192_96_python_FAM.py --fcount {fcount} --chip-dim {ifc} --in-csv {csv_file} --in-layout {layout_file} --timepoints {count_tp} --out-dir {out_folder} <br> Any other info you want to display?',
    'font': {'size': 20, 'color': 'black'},
    'showarrow': False
}

# Update the fig with annotations
fig.update_layout({'annotations': [details]})
fig.write_html(f'{out_folder}/{exp_name}_{instrument_type}_merged.html')