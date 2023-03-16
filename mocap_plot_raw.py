import pandas as pd
import math
import sys, re, os
import matplotlib.pyplot as mp
import matplotlib as mpl
import matplotlib.dates as dates
import numpy as np
import datetime
from matplotlib.colors import Normalize
from matplotlib import cm
import argparse
from MoCap.File import MoCapReader

# ============================================================================
# Plots the "raw" .tsv sensor data from the MoCap system.
# Optionally resamples the data, saves each sensor stream in a
# separate file. X, Y and Z coordinates can be put in the same plot with
# the -c parameter.
# The -w creates a 200 Hz 16bit int wave file which can be loaded
# into Elan for a visualisation of the movement.
#
# Example:
#   python mocap_plot_raw.py -c -f dyad_brainstorm_1_light_f1_50447_ev.tsv -F x_LHand -F x_RHand 
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--filename",
                     help="MoCap tsv file." )
parser.add_argument( "-F", "--filter", action="append",
                     help="Regexp to filter sensor name.", default=[] )
parser.add_argument( "-r", "--resample", default=None, type=str,
                     help="Resample time series (eg 50ms)." )
parser.add_argument( "-s", "--save", action="store_true",
                     help="Save each sensor in a separate file." )
parser.add_argument( "-w", "--wave", action="store_true",
                     help="Save each sensor as a pseudo wave file." )
parser.add_argument( "-c", "--combine", action="store_true",
                     help="Combine X, Y and Z in one plot." )
args = parser.parse_args()

# ============================================================================
# Plot functions
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
        if num_plots == 1: # We can't index if nrows==1
            ax = axes
        else:
            ax = axes[i]
        # Two plots, one for the zeroes and one for the rest.
        zeroes   = np.ma.masked_where(a_df[sensor].values == 0, a_df[sensor].values)
        nozeroes = np.ma.masked_where(a_df[sensor].values != 0, a_df[sensor].values)
        ax.plot(
            a_df.index.values,
            zeroes
        )
        ax.plot(
            a_df.index.values,
            nozeroes
        )
        #ax.vlines(a_df.index.values, 0, a_df[sensor].values) 
        ax.set_title( str(sensor) )
        #mp.locator_params(axis='x', nbins=20)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
        ax.xaxis.set_major_formatter(time_formatter)
        fig.autofmt_xdate()
    fig.tight_layout()

def plot_triplet(sensors, dfs, title=None):
    num_plots = len(sensors)
    fig, axes = mp.subplots(nrows=num_plots, ncols=1, figsize=(12,9), sharex=True) #, sharey=True)
    if title:
        fig.suptitle( title )
    for i in range(len(sensors)):
        if num_plots == 1: # We can't index if nrows==1
            ax = axes
        else:
            ax = axes[i]
        # Two plots, one for the zeroes and one for the rest.
        sensor = sensors[i]
        a_df = dfs[i]
        zeroes   = np.ma.masked_where(a_df[sensor].values == 0, a_df[sensor].values)
        nozeroes = np.ma.masked_where(a_df[sensor].values != 0, a_df[sensor].values)
        ax.plot(
            a_df.index.values,
            zeroes
        )
        ax.plot(
            a_df.index.values,
            nozeroes
        )
        #ax.vlines(a_df.index.values, 0, a_df[sensor].values) 
        ax.set_title( str(sensor) )
        #mp.locator_params(axis='x', nbins=20)
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
            a_df.index.values,
            a_df[l_group[i]].values,
            'go-', linewidth=0, markersize=1
            #'tab:green'
        )
        axes[i, 0].set_title(l_group[i])
        axes[i, 1].plot(
            a_df.index.values,
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
        print( sensor )
        axes.plot(
            a_df.index.values,
            a_df[sensor].values,
            label=str(sensor)
        )
    axes.legend(loc="upper right", fontsize=8)
    axes.xaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
    axes.xaxis.set_major_formatter(time_formatter)
    fig.autofmt_xdate()
    fig.tight_layout()

# All sensors in the same plot, stacked (e g _X, _Y, _Z)
def plot_group_combined_stacked(a_group, a_df, title=None):
    num_plots = len(a_group)
    if num_plots == 1:
        print( "Please rujn without the \"-c\" option." )
        sys.exit(1)
    fig, axes = mp.subplots(nrows=num_plots, ncols=1, figsize=(12,9), sharex=True, sharey=True)
    if title:
        fig.suptitle( title )
    for i in range(0, num_plots):
        zeroes   = np.ma.masked_where(a_df[a_group[i]].values == 0, a_df[a_group[i]].values)
        nozeroes = np.ma.masked_where(a_df[a_group[i]].values != 0, a_df[a_group[i]].values)
        #axes[i].vlines(a_df.index.values,
        #               0, a_df[a_group[i]].values) 
        axes[i].plot(
            a_df.index.values,
            #a_df[a_group[i]].values,
            zeroes,
            label=str(a_group[i])
        )
        axes[i].plot(
            a_df.index.values,
            #a_df[a_group[i]].values,
            nozeroes,
            label=str(a_group[i])
        )
        axes[i].set_title(a_group[i])
    axes[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
    axes[0].xaxis.set_major_formatter(time_formatter)
    fig.autofmt_xdate()
    fig.tight_layout()

# All sensors from two similar dataframes, one up, one down.
def plot_groups_combined_stacked(l_group, r_group, a_df, title=None, subtitles=None):
    fig, axes = mp.subplots(nrows=2, ncols=1, figsize=(12,6), sharex=True, sharey=True)
    if title:
        fig.suptitle( title )
    for sensor in l_group:
        axes[0].plot(
            a_df.index.values,
            a_df[sensor].values,
            label=str(sensor)
        )
    axes[0].legend(loc="upper right", fontsize=8)
    for sensor in r_group:
        axes[1].plot(
            a_df.index.values,
            a_df[sensor].values,
            label=str(sensor)
        )
    axes[1].legend(loc="upper right", fontsize=8)
    if subtitles:
        for i, subtitle in enumerate(subtitles):
            axes[i].set_title( subtitles[i] )
    fig.tight_layout()

# ============================================================================
# Main
# ============================================================================

r = MoCapReader()
res = r.read( args.filename )
if not res:
    print( "Error reading", args.filename )
    sys.exit(1)

#print( r.get_info() )
df_pos = r.get_df()
df_dis = r.get_df_dist()
df_vel = r.get_df_vel()
df_acc = r.get_df_acc()
r.save(index=True)
print( df_pos.head() )

# ============================================================================
# Apply the filters.
# ============================================================================

filtered_columns = []
for sensor in df_pos.columns:
    for filter_re in args.filter:
        if re.search( filter_re, sensor ):
            filtered_columns.append( sensor )
if len(filtered_columns) == 0: # If none, take all!
    filtered_columns = df_pos.columns
df_pos = df_pos[filtered_columns]# Not necessary...
print( df_pos.head() )
print( df_pos.tail() )

# ============================================================================
# Print the number of zero values.
# ============================================================================

print( "\nCount Zero:" )
zeroes = r.count_zero()
#print( zeroes )
for col in zeroes:
    count, frames = zeroes[col]
    if count > 0:
        pct = count * 100.0 / frames 
        print( f"{col:<20} {count:8d} ({pct:5.1f}%)" )

# ============================================================================
# Resampling.
# ============================================================================

# sum() works better than mean() or max()
if args.resample:
    print( "Resampling", args.resample )
    df_pos = df_pos.resample(args.resample).sum()
    print( "\n============ After resampling =================" )
    print( df_pos.head() )
    print( df_pos.tail() )
    df_dis = df_dis.resample(args.resample).sum()
    print( "\n============ After resampling =================" )
    print( df_dis.head() )
    print( df_dis.tail() )
    df_vel = df_vel.resample(args.resample).sum()
    print( "\n============ After resampling =================" )
    print( df_vel.head() )
    print( df_vel.tail() )
    df_acc = df_acc.resample(args.resample).sum()
    print( "\n============ After resampling =================" )
    print( df_acc.head() )
    print( df_acc.tail() )

# ============================================================================
# Saving.
# ============================================================================

if args.save or args.wave:
    import scipy.io.wavfile as wv
    filepath_bits = os.path.split( args.filename )
    for col in filtered_columns:
        data = df_pos[col]
        if args.save:
            col_filename = os.path.join(filepath_bits[0],
                                        filepath_bits[1][:-4] + "_" + col + ".tsv" )
            print( f"Saving {col_filename}" )
            data.to_csv(col_filename,
                               sep='\t',
                               index=False)
        if args.wave:
            wav_filename = os.path.join(filepath_bits[0],
                                        filepath_bits[1][:-4] + "_" + col + ".wav" )
            print( f"Saving {wav_filename}" )
            #data = (data - data.mean())/data.std() # z-scaling
            data = 2 * (data-data.min())/(data.max()-data.min())-1.0
            data = data * 32000
            wv.write(wav_filename, 200, data.astype(np.int16)) #astype(np.float32))

# ============================================================================
# Plots.
# ============================================================================

if not args.combine:
    for col in filtered_columns:
        plot_group([col], df_pos)
else:
    for i in range(0, len(filtered_columns), 3): # triples of _x _y _z
        cols = filtered_columns[i:i+3]
        #plot_group_combined(cols, df_pos, title=None) # In the same plot
        plot_group_combined_stacked(cols, df_pos, title=None)
        col_name = filtered_columns[i][:-2] # Remove _X
        plot_triplet( [col_name+"_d3D", col_name+"_vel", col_name+"_acc"],
                      [df_dis, df_vel.abs(), df_acc.abs()] )

mp.show()
