import pandas as pd
import math
import sys
import matplotlib.pyplot as mp
import matplotlib as mpl
import matplotlib.dates as dates
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
import argparse

# Use PYVENV in Development
# (PYVENV) pberck@ip30-163 MoCap %
# python mocap_plot.py -d mocap_valentijn/beach_repr_2b_dists.tsv

# Create/add to an (existing) EAF file.

# Resample!

# ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--distsfilename",
                     help="MoCap tsv file (distances, from mocap_gen_dists.py)." )
parser.add_argument( "-F", "--filter", default = "_",
                     help="Filter column names." )
parser.add_argument( "-r", "--resample", default=None, type=str,
                     help="Resample time series." )
args = parser.parse_args()

# ----------------------------

# Each sensor in a separate plot.
def plot_group(a_group, a_df, title=None):
    num_plots = len(a_group)
    fig, axes = mp.subplots(nrows=num_plots, ncols=1, figsize=(12,6), sharex=True, sharey=True)
    if title:
        fig.suptitle( title )
    for i, sensor in enumerate(a_group):
        if num_plots == 1: # we can't index if nrows==1
            ax = axes
        else:
            ax = axes[i]
        # Two plots, one for the zeroes and one for the rest. Initially
        # we had just one plot with a_df[sensor].values.
        zeroes = np.ma.masked_where(a_df[sensor].values == 0, a_df[sensor].values)
        nozeroes = np.ma.masked_where(a_df[sensor].values != 0, a_df[sensor].values)
        if False:
            ax.plot(
                a_df["Timestamp"].values,
                #a_df[sensor].values,
                zeroes
            )
            ax.plot(
                a_df["Timestamp"].values,
                #a_df[sensor].values,
                nozeroes
            )
        else:
            ax.vlines(a_df["Timestamp"].values,
                      0, a_df[sensor].values) 
        ax.set_title( str(sensor) )
    fig.tight_layout()

# Similar dataframes, one left, one right.
def plot_groups_lr(l_group, r_group, a_df, title=None):
    num_plots = len(l_group) # assume same length
    fig, axes = mp.subplots(nrows=num_plots, ncols=2, figsize=(12,6), sharex=True, sharey=True)
    if title:
        fig.suptitle( title )
    for i in range(0, num_plots):
        axes[i, 0].plot(
            a_df["Timestamp"].values,
            a_df[l_group[i]].values,
            'go-', linewidth=0, markersize=1
            #'tab:green'
        )
        axes[i, 0].set_title(l_group[i])
        axes[i, 1].plot(
            a_df["Timestamp"].values,
            a_df[r_group[i]].values,
            'co-', linewidth=0, markersize=1
            #'tab:cyan'
        )
        axes[i, 1].set_title(r_group[i])
    fig.tight_layout()

# All sensors in the same plot.
def plot_group_combined(a_group, a_df, title=None):
    fig, axes = mp.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=True, sharey=True)
    if title:
        fig.suptitle( title )
    for sensor in a_group:
        axes.plot(
            a_df["Timestamp"].values,
            a_df[sensor].values,
            label=str(sensor)
        )
    axes.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

# All sensors from two similar dataframes, one up, one down.
def plot_groups_combined_stacked(l_group, r_group, a_df, title=None, subtitles=None):
    fig, axes = mp.subplots(nrows=2, ncols=1, figsize=(12,6), sharex=True, sharey=True)
    if title:
        fig.suptitle( title )
    for sensor in l_group:
        axes[0].plot(
            a_df["Timestamp"].values,
            a_df[sensor].values,
            label=str(sensor)
        )
    axes[0].legend(loc="upper right", fontsize=8)
    for sensor in r_group:
        axes[1].plot(
            a_df["Timestamp"].values,
            a_df[sensor].values,
            label=str(sensor)
        )
    axes[1].legend(loc="upper right", fontsize=8)
    if subtitles:
        for i, subtitle in enumerate(subtitles):
            axes[i].set_title( subtitles[i] )
    fig.tight_layout()

# ----------------------------

# Read the distance data
df_dists = pd.read_csv(
    args.distsfilename,
    sep="\t"
)
print( df_dists.columns )
filtered_columns = [ col for col in df_dists.columns if args.filter in col ]
print( filtered_columns )

# ----------------------------

# A few ad hoc distance groups.
group_Head = ["x_HeadFront", "x_HeadL", "x_HeadR", "x_HeadTop"]

group_LFoot = ["x_LAnkleOut", "x_LForefootIn", "x_LForefootOut", "x_LHeelBack", "x_LKneeOut",
               "x_LShin", "x_LThigh", "x_LToeTip"]
group_RFoot = ["x_RAnkleOut", "x_RForefootIn", "x_RForefootOut", "x_RHeelBack", "x_RKneeOut",
               "x_RShin", "x_RThigh", "x_RToeTip"]

group_LArm = ["x_LShoulderBack", "x_LShoulderTop", "x_LArm", "x_LElbowOut", "x_LHandIn",
              "x_LHandOut", "x_LWristIn", "x_LWristOut" ]
group_RArm = ["x_RShoulderBack", "x_RShoulderTop", "x_RArm", "x_RElbowOut", "x_RHandIn",
              "x_RHandOut", "x_RWristIn", "x_RWristOut" ]

group_Body = ["x_BackL", "x_BackR", "x_Chest", "x_SpineTop", 
              "x_WaistLBack", "x_WaistLFront", "x_WaistRBack", "x_WaistRFront"]

# RESAMPLING

df_dists['td'] = pd.to_timedelta(df_dists['Timestamp'], 's') # Create a timedelta column
df_dists = df_dists.set_index(df_dists['td']) # and use it as index
print( df_dists.head() )

# sum() works better than mean() or max()
if args.resample:
    print( "Resampling", args.resample )
    df_dists = df_dists.resample(args.resample).sum() 
    print( df_dists.head() )
    print( df_dists.tail() )
    df_dists = df_dists[:-1] # To remove the last invalid "peak" (wrong timestamp?)
    
for col in filtered_columns:
    plot_group([col], df_dists)

#plot_groups_lr( group_LArm, group_RArm, df_dists, title="Left and Right Arm" )

# Create a new dataframe with "distance moved across threshold" indicators.
# Determine threshold through statistical analysis?
df_dists_t = pd.DataFrame()
df_dists_t["Timestamp"] = df_dists["Timestamp"]

'''
for sensor in group_LArm + group_RArm:
    df_dists_t[sensor+'_T'] = np.where( df_dists[sensor] > 1, 3, 0 )
print( df_dists_t )

# Plot distances
fig, axes = mp.subplots(nrows=2, ncols=1, figsize=(12,6), sharex=True, sharey=True)
fig.suptitle( "Distances Right and Left Elbows" )

# Field to test colour/size selection in plot.
col = np.where(df_dists_t["x_LElbowOut_T"] > 1, 'r', 'b')
siz = np.where(df_dists_t["x_LElbowOut_T"] > 1, 1, 0)
cmap, norm = mpl.colors.from_levels_and_colors([0, 100, 1000], ['r', 'k'])

axes[0].plot(
    df_dists["Timestamp"].values,
    df_dists["x_LElbowOut"].values,
)
axes[0].set_title("x_LElbowOut and marker")
axes[0].scatter(
    df_dists["Timestamp"].values,
    df_dists_t["x_LElbowOut_T"].values,
    s=siz, #c=col,
    cmap='viridis'
)
axes[0].legend(loc="upper right")

#axes[0].vlines(df_dists["Timestamp"].values,
#               0, 2*df_dists_t["x_LElbowOut_T"].values) # colour according to real value?

axes[0].scatter(
    df_dists["Timestamp"].values,
    df_dists_t["x_LElbowOut_T"].values,
    s=siz, c=col,
    #c=df_dists["x_LElbowOut"].values, cmap='viridis'
)

my_cmap = mp.get_cmap("viridis")
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
axes[1].bar(
    df_dists["Timestamp"].values,
    df_dists_t["x_LElbowOut_T"].values,
    color=my_cmap( rescale(df_dists["x_LElbowOut"].values) )
    #s=siz, c=col,
    #c=df_dists["x_LElbowOut"].values, cmap='viridis'
)
    
axes[1].legend(loc="upper right")
fig.tight_layout()
'''

mp.show()
