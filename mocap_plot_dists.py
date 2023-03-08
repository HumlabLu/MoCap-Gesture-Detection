import pandas as pd
import math
import sys, re
import matplotlib.pyplot as mp
import matplotlib as mpl
import matplotlib.dates as dates
import numpy as np
import datetime
from matplotlib.colors import Normalize
from matplotlib import cm
import argparse

# Use PYVENV in Development
# (PYVENV) pberck@ip30-163 MoCap %

# ============================================================================
# python mocap_plot.py -d mocap_valentijn/beach_repr_2b_dists.tsv
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--distsfilename",
                     help="MoCap tsv file (distances, from mocap_gen_dists.py)." )
parser.add_argument( "-F", "--filter", action="append",
                     help="Regexp to filter sensor name.", default=[] )
parser.add_argument( "-r", "--resample", default=None, type=str,
                     help="Resample time series." )
args = parser.parse_args()

# ============================================================================

def time_ticks(x, pos):
    x = x / 1_000_000 # Scale back to milliseconds, pd.Timedelta.resolution ) = 0 days 00:00:00.000000001
    d = datetime.timedelta(milliseconds=x)
    d_str = str(d)
    return str(d)[0:11]
time_formatter = mpl.ticker.FuncFormatter(time_ticks)

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
                #a_df["Timestamp"].values,
                a_df.index.values,
                #a_df[sensor].values,
                zeroes
            )
            ax.plot(
                #a_df["Timestamp"].values,
                a_df.index.values,
                #a_df[sensor].values,
                nozeroes
            )
        else:
            ax.vlines(a_df.index.values,
                      0, a_df[sensor].values) 
        ax.set_title( str(sensor) )
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
        ax.xaxis.set_major_formatter(time_formatter)
        fig.autofmt_xdate()
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

# ============================================================================
# Read the data.
# ============================================================================

df_dists = pd.read_csv(
    args.distsfilename,
    sep="\t"
)
print( df_dists.columns )

# ============================================================================
# Filter the coilumns
# ============================================================================

filtered_columns = []
for filter_re in args.filter:
    [ filtered_columns.append(s)
      for s in df_dists.columns if re.search(filter_re, s) and s not in filtered_columns ]
if len(filtered_columns) == 0: # If only empty then take all.
    filtered_columns = df_dists.columns
print( df_dists.head() )
print( df_dists.tail() )

# ============================================================================
# Add timedelta column as index.
# ============================================================================

df_dists['td'] = pd.to_timedelta(df_dists['Timestamp'], 's') # Create a timedelta column
df_dists = df_dists.set_index(df_dists['td']) # and use it as index
print( df_dists.head() )

# ============================================================================
# Resample
# ============================================================================

# sum() works better than mean() or max()
if args.resample:
    print( "Resampling", args.resample )
    df_dists = df_dists.resample(args.resample).sum() 
    print( df_dists.head() )
    print( df_dists.tail() )
    #df_dists = df_dists[:-1] # To remove the last invalid "peak" (wrong timestamp?)

# ============================================================================
# Plot
# ============================================================================

for col in filtered_columns:
    plot_group([col], df_dists)

mp.show()
