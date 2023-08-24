#!/usr/bin/env python

import pandas as pd
import numpy as np
from os import listdir,path
import warnings
import math
import csv
import argparse
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit
 
ifc = 192
fcount = 1691383040
tgap = 1
instrument_type = 'BM'

if instrument_type == 'EP':
      count_tp = 1
else:
     count_tp = 13
 
#print(count_tp)

exp_name = str(fcount) + '_' + str(ifc)
 
time_assign = {}
for cycle in range(1,38):
      tpoint = "t" + str(cycle)
      time_assign[tpoint] = tgap + 3 + (cycle-1) * 5
 
#print(time_assign)

def gettime(tname):
      realt = time_assign[tname]
      return (realt) 
 
#print(gettime('t1'))

layout_file = '192_assignment.xlsx'
#print(layout_file)

csv_file = '1691383040.csv'
#print(csv_file)
 
out_folder = 'output'
#print(out_folder)


if path.exists(layout_file) == False:
      raise Exception((layout_file + " doesn't exist"))
 
#print(path.exists(layout_file))

if path.exists(csv_file) == False:
      raise Exception((csv_file + " doesn't exist"))
 
#print(path.exists(csv_file))


probe_df = pd.read_csv(csv_file, sep=",", header = 9238, nrows = 4608, skip_blank_lines=False)
#print(probe_df)

reference_df = pd.read_csv(csv_file, sep=",", header = 4627, nrows = 4608, skip_blank_lines=False)
#print(reference_df)

bkgd_ref_df = pd.read_csv(csv_file, sep=",", header = 13849, nrows = 4608, skip_blank_lines=False)
#print(bkgd_ref_df)

bkgd_probe_df = pd.read_csv(csv_file, sep=",", header = 18460, nrows = 4608, skip_blank_lines=False)
#print(bkgd_probe_df)


c_to_drop = 'Unnamed: ' + str(count_tp+1)
probe_df = probe_df.set_index("Chamber ID").drop(c_to_drop, axis = 1)
reference_df = reference_df.set_index("Chamber ID").drop(c_to_drop, axis = 1)
bkgd_ref_df = bkgd_ref_df.set_index("Chamber ID").drop(c_to_drop, axis = 1)
bkgd_probe_df = bkgd_probe_df.set_index("Chamber ID").drop(c_to_drop, axis = 1)
 
#print(probe_df.columns)
probe_df.columns = probe_df.columns.str.lstrip()
#print(probe_df.columns) 
reference_df.columns = reference_df.columns.str.lstrip()
bkgd_ref_df.columns = bkgd_ref_df.columns.str.lstrip()
bkgd_probe_df.columns = bkgd_probe_df.columns.str.lstrip()
 

#print(probe_df.columns)
probe_df.columns = ['t' + str(col) for col in probe_df.columns]
#print(probe_df.columns)
 
reference_df.columns = ['t' + str(col) for col in reference_df.columns]
bkgd_ref_df.columns = ['t' + str(col) for col in bkgd_ref_df.columns]
bkgd_probe_df.columns = ['t' + str(col) for col in bkgd_probe_df.columns]


probe_bkgd_substracted = probe_df.subtract(bkgd_probe_df)
#print(probe_bkgd_substracted)
ref_bkgd_substracted = reference_df.subtract(bkgd_ref_df)
#print(ref_bkgd_substracted)

signal_df = pd.DataFrame(probe_bkgd_substracted/ref_bkgd_substracted)
#print(signal_df)

signal_df = signal_df.reset_index()
#print(signal_df)

splitassignment = signal_df['Chamber ID'].str.split("-",n=1,expand=True)
#print(splitassignment)

signal_df["sampleID"] = splitassignment[0]
#print(signal_df["sampleID"])
signal_df["assayID"] = splitassignment[1]
#print(signal_df["assayID"])

signal_df = signal_df.set_index('Chamber ID')
#print(signal_df)

sampleID_list = signal_df.sampleID.unique()
#print(sampleID_list)

assayID_list = signal_df.assayID.unique()
#print(assayID_list)


signal_out_csv_1 = f"{out_folder}/{exp_name}_{instrument_type}_1_signal_bkgdsubtracted_norm_{str(count_tp)}.csv"
#print(signal_out_csv_1)
signal_df.to_csv(signal_out_csv_1)
 

samples_layout_wo_string = pd.read_excel(f"{layout_file}",sheet_name='layout_samples', engine="openpyxl")
#print(samples_layout_wo_string)
samples_layout = samples_layout_wo_string.applymap(str)
#print(samples_layout)


assays_layout_wo_string = pd.read_excel(f"{layout_file}",sheet_name='layout_assays', engine='openpyxl')
assays_layout = assays_layout_wo_string.applymap(str)
#print(assays_layout)


assays = pd.read_excel(f"{layout_file}",sheet_name='assays', engine='openpyxl')
#print(assays)
assays_zip = zip(assays_layout.values.reshape(-1),assays.values.reshape(-1))
#print(assays_zip)
assays_dict = dict(assays_zip)
#print(assays_dict)


samples = pd.read_excel(path.join('',layout_file),sheet_name='samples', engine='openpyxl')
#print(samples)
samples_zip = zip(samples_layout.values.reshape(-1),samples.values.reshape(-1))
samples_dict = dict(samples_zip)
# print(samples_dict)


# print(signal_df['assayID'])
signal_df['assay'] = signal_df['assayID'].map(assays_dict)
# print(signal_df['assay'])


# print(signal_df['sampleID'])
signal_df['sample'] = signal_df['sampleID'].map(samples_dict)
# print(signal_df['sample'])


signal_out_csv_2 = f"{out_folder}/ {exp_name}_{instrument_type}_2_signal_bkgdsubtracted_norm_named_{str(count_tp)}.csv"
# print(signal_out_csv_2)
signal_df.to_csv(signal_out_csv_2)
 

 
t_names = []
for i in range(1,count_tp+1):
      t_names.append(('t' + str(i)))
 
# print(t_names)

 
if ifc == '96':
      new_array = np.stack(assays[['C1','C2','C3','C4','C5']].values,axis=-1)
else:
      new_array = np.stack(assays[['C1','C2', 'C3']].values,axis=-1)
 
# print(new_array)
assay_list = np.concatenate(new_array).tolist()
print('identified crRNAs: ',len(assay_list))


 
if ifc == '96':
      new_array = np.stack(samples[['C1','C2','C3','C4','C5','C6','C7', 'C8','C9','C10','C11','C12']].values,axis=-1)
else:
      new_array = np.stack(samples[['C1','C2','C3','C4','C5','C6', 'C7','C8','C9','C10','C11','C12', 'C13','C14','C15','C16','C17','C18', 'C19','C20','C21','C22','C23','C24']].values,axis=-1)
 
# print(new_array)
sample_list = np.concatenate(new_array).tolist()
print('identified samples: ',len(sample_list))
 


 
grouped_stuff = signal_df.groupby(['assay','sample'])
medians = grouped_stuff.median(numeric_only = True)
# print(medians)
 


 
med_frames = {}
for name in t_names:
      time_med = medians[name].unstack()
      time_med.index.names=['']
      time_med.columns.names=['']
      med_frames[name] = time_med
 
#print(med_frames)



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

def plt_heatmap(df_dict, samplelist, assaylist, tp):    
    frame = df_dict[tp][samplelist].reindex(assaylist)
    fig, axes = plt.subplots(1,1,figsize=(len(frame.columns.values)*0.5,len(frame.index.values)*0.5))
    ax = sns.heatmap(frame,cmap='Reds',square = True,cbar_kws={'pad':0.002}, annot_kws={"size": 20})
    rt = gettime(tp)
    plt.title(exp_name+' '+str(rt)+'min - median values', size = 28)
    plt.xlabel('Samples', size = 14)
    plt.ylabel('Assays', size = 14)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)
    plt.yticks(rotation=0) 

    tgt_num = len(sample_list)
    gd_num = len(assay_list)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    h_lines = np.arange(3,gd_num,3)
    v_lines = np.arange(3,tgt_num,3)
    axes.hlines(h_lines, colors = 'silver',alpha=0.9,linewidths = 0.35,*axes.get_xlim())
    axes.vlines(v_lines, colors = 'silver',alpha=0.9,linewidths = 0.35,*axes.get_ylim())

    #plt.savefig(path.join(out_folder, exp_name + '_'+instrument_type +'2_heatmap_'+str(tp)+'.png'), format='png', bbox_inches='tight', dpi=400)
    plt.savefig(f'{out_folder}/{exp_name}_{instrument_type}2_heatmap_{str(tp)}.png', format='png', bbox_inches='tight', dpi=400)

plt_heatmap(med_frames, sample_list, assay_list, 't13')                     