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

# Read the sensor data into a dataframe.
df      = None
df_rows = []
lnum    = 0
freq    = 200 # Parse this from file header.
with open(args.filename, "r") as f:
    for line in f:
        bits = line.split()
        if bits[0] == "FREQUENCY":
            freq = int(bits[1])
        if bits[0] == "MARKER_NAMES":
            column_names = bits[1:] # We add a Timestamp later to this too.
            print( column_names )
            new_column_names = ["Timestamp"]
            for col in column_names:
                # We have three values for each sensor.
                for coord in ["_X", "_Y", "_Z"]: 
                    new_column_names.append( col+coord )
        if len(bits) > 65:
            try:
                bits     = [ float(x) for x in bits ]
                triplets = [bits[i:i + 3] for i in range(2,len(bits)-2, 3)]
                df_rows.append( bits[1:] ) # Skip index number.
            except ValueError:
                print( "Skipping line", lnum )
        lnum += 1

df_pos = pd.DataFrame(
    df_rows,
    columns=new_column_names
)

# ============================================================================
# Apply the filters.
# ============================================================================

filtered_columns = ["Timestamp"]
for filter_re in args.filter:
    [ filtered_columns.append(s)
      for s in df_pos.columns if re.search(filter_re, s) and s not in filtered_columns ]
if len(filtered_columns) == 1: # If only "Timestamp" then take all.
    filtered_columns = df_pos.columns
df_pos = df_pos[filtered_columns] # Not necessary, might save some memory.
print( df_pos.head() )
print( df_pos.tail() )

# ============================================================================
# Print the number of zero values.
# ============================================================================

print( "\nCount Zero:" )
for col in df_pos.columns[1:]:
    count = (df_pos[col] == 0).sum()
    frames = len(df_pos[col])
    pct = count * 100.0 / frames 
    print( f"{col:<20} {count:8d} ({pct:5.1f}%)" )
    
# ============================================================================
# Resampling.
# ============================================================================

df_pos['td'] = pd.to_timedelta(df_pos['Timestamp'], 's') # Create a Timestamp column
df_pos = df_pos.set_index(df_pos['td']) # and use it as index
#df_pos = df_pos.drop(["Timestamp"], axis=1)

# sum() works better than mean() or max()
if args.resample:
    print( "Resampling", args.resample )
    df_pos = df_pos.resample(args.resample).sum() # or mean()
    print( df_pos.head() )
    print( df_pos.tail() )
    print( "\n============ After resampling =================" )
    print( df_pos.head() )
    print( df_pos.tail() )

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
    for col in filtered_columns[1:]: # Skip "Timestamp"
        plot_group([col], df_pos)
else:
    for i in range(0, len(filtered_columns[1:]), 3): # triples of _x _y _z
        cols = filtered_columns[i+1:i+1+3] # +1 to skip "Timestamp"
        #plot_group_combined(cols, df_pos, title=None) # In the same plot
        plot_group_combined_stacked(cols, df_pos, title=None)
    
#df_pos = (df_pos - df_pos.mean())/df_pos.std() # Normalisation
#df_pos = (df_pos - df_pos.min())/(df_pos.max()-df_pos.min()) # Min-max normalisation
#for col in filtered_columns[1:]: # Skip "Timestamp"
#    plot_group([col], df_pos)

mp.show()
