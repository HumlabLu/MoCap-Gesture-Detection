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
import pympi

# Use PYVENV in Development
# (PYVENV) pberck@ip30-163 MoCap %
# python mocap_03.py -f mocap_valentijn/beach_repr_2b_velocity_M.tsv -d mocap_valentijn/beach_repr_2b_dists.tsv

# Create/add to an (existing) EAF file.

# Resample!

# ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--filename",
                     help="MoCap tsv file (velocities)." )
parser.add_argument( "-d", "--distsfilename",
                     help="MoCap tsv file (distances, from mocap_gen_distances.py)." )
parser.add_argument( "-e", "--eaffilename",
                     help="EAF file to augment." )
args = parser.parse_args()

'''
ls mocap_valentijn/*tsv
  mocap_valentijn/beach_repr_2b.tsv		mocap_valentijn/beach_repr_2b_ang_acc_RH.tsv
  mocap_valentijn/beach_repr_2b_ang_acc_LH.tsv	mocap_valentijn/beach_repr_2b_velocity_M.tsv

ls *tsv
  beach_repr_2b.tsv		beach_repr_2b_ang_acc_RH.tsv	beach_repr_2b_velocity_M.tsv
  beach_repr_2b_ang_acc_LH.tsv	beach_repr_2b_dists.tsv
'''

# ----------------------------

# Each sensor in a separate plot.
def plot_group(a_group, a_df, title=None):
    num_plots = len(a_group)
    fig, axes = mp.subplots(nrows=num_plots, ncols=1, figsize=(12,6), sharex=True, sharey=True)
    if title:
        fig.suptitle( title )
    for i, sensor in enumerate(a_group):
        axes[i].plot(
            a_df["Timestamp"].values,
            a_df[sensor].values
        )
        axes[i].set_title( str(sensor) )
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


'''
(PYVENV) pberck@ip30-163 MoCap % head beach_repr_2b_velocity_M.tsv
NO_OF_FRAMES	20887
NO_OF_DATA_TYPES	28
FREQUENCY	200
TIME_STAMP	2022-11-22, 21:34:11
DATA_INCLUDED	Velocity
DATA_TYPES	x_LWristOut_vel_M	X_LWristIn_vel_M	x_LHandOut_vel_M	x_LHandIn_vel_M	x_RWristOut_vel_M	x_RWristIn_vel_M	x_RHandOut_vel_M	x_RHandIn_vel_M	x_RThumb1_vel_M	x_RThumbTip_vel_M	x_RIndex2_vel_M	x_RIndexTip_vel_M	x_RMiddle2_vel_M	x_RMiddleTip_vel_M	x_RRing2_vel_M	x_RRingTip_vel_M	x_RPinky2_vel_M	x_RPinkyTip_vel_M	x_LThumb1_vel_M	x_LThumbTip_vel_M	x_LIndex2_vel_M	x_LIndexTip_vel_M	x_LMiddle2_vel_M	x_LMiddleTip_vel_M	x_LRing2_vel_M	x_LRingTip_vel_M	x_LPinky2_vel_M	x_LPinkyTip_vel_M


1	0.00000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000
2	0.00500	12.438	14.483	73.739	109.118	12.927	11.315	11.895	12.532	14.295	14.165	73.827	265.273	191.557	22.159	16.281	16.797	19.674	26.238	161.599	184.912	109.435	119.367	142.256	161.760	164.861	208.077	124.859	144.613
'''

# Read velocities file.
df      = None
df_rows = []
lnum    = 0
freq    = 200 # Get from file header.
with open(args.filename, "r") as f:
    for line in f:
        bits = line.split()
        #print( lnum, len(bits) )
        if len(bits) > 1:
            if bits[0] == "FREQUENCY":
                freq = int(bits[1])
            if bits[0] == "DATA_TYPES":
                column_names = bits # We add a Timestamp later to this one too.
                print( column_names )
        if len(bits) > 15 and lnum > 7:
            bits = [ float(x) for x in bits ]
            df_rows.append( bits[1:] ) #skip index number
        lnum += 1

# Change the name of the first column to Timestamp.
column_names[0] = "Timestamp"

# Create the dataframe.
df = pd.DataFrame(
    df_rows,
    columns=column_names
)
print( df.head() )
#df['Time'] = pd.to_datetime(df['Timestamp']) # not used

# Read the distance data
df_dists = pd.read_csv(
    args.distsfilename,
    sep="\t"
)

#df['x_LWristOut_vel_M_T'] = np.where( df["x_LWristOut_vel_M"] > 240, 240, 0 )

# ----------------------------

#for x in column_names:
#    print( x )
# check for "finger movement only", "hand movement", "arm movement" (not in this data, use distances?)

group_LHand_M    = ["x_LWristOut_vel_M", "x_LWristIn_vel_M", "x_LHandOut_vel_M", "x_LHandIn_vel_M"]

group_LFingers_M = ["x_LThumb1_vel_M", "x_LThumbTip_vel_M", "x_LIndex2_vel_M", "x_LIndexTip_vel_M",
                    "x_LMiddle2_vel_M", "x_LMiddleTip_vel_M", "x_LRing2_vel_M", "x_LRingTip_vel_M",
                    "x_LPinky2_vel_M", "x_LPinkyTip_vel_M"]

group_RHand_M    = ["X_RWristOut_vel_M", "x_RWristIn_vel_M", "x_RHandOut_vel_M", "x_RHandIn_vel_M"]

group_RFingers_M = ["x_RThumb1_vel_M", "x_RThumbTip_vel_M", "x_RIndex2_vel_M", "x_RIndexTip_vel_M",
                    "x_RMiddle2_vel_M", "x_RMiddleTip_vel_M", "x_RRing2_vel_M", "x_RRingTip_vel_M",
                    "x_RPinky2_vel_M", "x_RPinkyTip_vel_M"]

'''
print( ",".join(sorted(df_dists.columns)) )
x_BackL, x_BackR, x_Chest, x_HeadFront, x_HeadL, x_HeadR, x_HeadTop, 

x_LAnkleOut, x_LArm, x_LElbowOut, x_LForefootIn, x_LForefootOut, x_LHandIn, x_LHandOut, x_LHeelBack, x_LIndex2, x_LIndexTip, x_LKneeOut, x_LMiddle2, x_LMiddleTip, x_LPinky2, x_LPinkyTip, x_LRing2, x_LRingTip, x_LShin, x_LShoulderBack, x_LShoulderTop, x_LThigh, x_LThumb1, x_LThumbTip, x_LToeTip, x_LWristIn, x_LWristOut, 

x_RAnkleOut, x_RArm, x_RElbowOut, x_RForefootIn, x_RForefootOut, x_RHandIn, x_RHandOut, x_RHeelBack, x_RIndex2, x_RIndexTip, x_RKneeOut, x_RMiddle2, x_RMiddleTip, x_RPinky2, x_RPinkyTip, x_RRing2, x_RRingTip, x_RShin, x_RShoulderBack, x_RShoulderTop, x_RThigh, x_RThumb1, x_RThumbTip, x_RToeTip, x_RWristIn, x_RWristOut, 

x_SpineTop, x_WaistLBack, x_WaistLFront, x_WaistRBack, x_WaistRFront
'''

# A few ad hoc distance groups.
group_Head = ["x_HeadFront", "x_HeadL", "x_HeadR", "x_HeadTop"]

group_LFoot = ["x_LAnkleOut", "x_LForefootIn", "x_LForefootOut", "x_LHeelBack", "x_LKneeOut",
               "x_LShin", "x_LThigh", "x_LToeTip"]
group_RFoot = ["x_RAnkleOut", "x_RForefootIn", "x_RForefootOut", "x_RHeelBack", "x_RKneeOut",
               "x_RShin", "x_RThigh", "x_RToeTip"]

group_LArm = ["X_LShoulderBack", "x_LShoulderTop", "x_LArm", "x_LElbowOut", "x_LHandIn",
              "x_LHandOut", "x_LWristIn", "x_LWristOut" ]
group_RArm = ["X_RShoulderBack", "x_RShoulderTop", "x_RArm", "x_RElbowOut", "x_RHandIn",
              "x_RHandOut", "x_RWristIn", "x_RWristOut" ]

group_Body = ["x_BackL", "x_BackR", "x_Chest", "x_SpineTop", 
              "x_WaistLBack", "x_WaistLFront", "X_WaistRBack", "x_WaistRFront"]


# RESAMPLING

df_dists['td'] = pd.to_timedelta(df_dists['Timestamp'], 's') # Create a timedelta column
df_dists = df_dists.set_index(df_dists['td']) # and use it as index
print( df_dists.head() )

# max() works better than mean()
df_dists = df_dists.resample("50ms").max() # This resamples the 200 Hz to 20 Hz
print( df_dists.head() )

# NAIVE IMPLEMENTATION OVER DISTANCE GROUPS

# maybe group the st/et's in a new trier, take the "max" extent over the columns to catch
# "group" movement -> resampling helps here!

# Take several sensors, put values in one timeseries to "collapse" to one dimension
# df['total']= df.iloc[:, -4:-1].sum(axis=1)
# col_list= list(df)
# col_list.remove('english')
# df['Sum'] = df[col_list].sum(axis=1)

df_dists['LATotal'] = df_dists[["x_LHandIn", "x_LHandOut", "x_LWristIn", "x_LWristOut"]].sum(axis=1)
df_dists['RATotal'] = df_dists[["x_RHandIn", "x_RHandOut", "x_RWristIn", "x_RWristOut"]].sum(axis=1)

if not args.eaffilename:
    print( "No EAF filename specified, quitting." )
    sys.exit(1)

eaf = pympi.Elan.Eaf(file_path=args.eaffilename, author='mocap_04.py')

for sensor in ["LATotal", "RATotal"]:
#for sensor in ["x_LHandIn", "x_LHandOut", "x_LWristIn", "x_LWristOut",
#               "x_RHandIn", "x_RHandOut", "x_RWristIn", "x_RWristOut",
#               "x_WaistLBack", "x_WaistRBack"]:# group_LArm+group_RArm:
    dist_max = df_dists[sensor].max()
    dist_min = df_dists[sensor].min()
    print( sensor, dist_min, dist_max )
    eaf.add_tier( sensor, ling='default-lt' )
    # instead of threshhold, difference in direction, we have that data?
    threshold = dist_max - (dist_max * 0.90) # take if > "10%"
    inside = False
    st = -1
    et = -1
    annotations = []
    previous_annotation = [0, 0]
    current_annotation = [0, 0]
    for ts, x in zip(df_dists["Timestamp"].values, df_dists[sensor].values):
    #for ts, x in zip(df_dists.index.values, df_dists[sensor].values): # timedeltas are microseconds
        if not inside and x > threshold:
            print( "NEW {:.3f} {:.4f}".format(float(ts), float(x)) )
            inside = True
            #st = int(ts / 1000000) # start time
            st = int(ts * 1000) # start time
            empty_time = st - previous_annotation[1] # to see if close to previous
            if empty_time < 200: #arbitrary... 120ms
                print( "Short" )
                print( " p ", previous_annotation )
                print( " c ", current_annotation )
                st = previous_annotation[0] # cheat, and put the previous start time
                annotations = annotations[:-1] # and remove previous annotation.
            # add to annotations here?
            current_annotation = [ st ] 
            empty = 0
            # concat annotations if close to gether? postprocess?
        elif not inside:
            pass
        elif inside and x <= threshold:
            print( "--- {:.3f} {:.4f}".format(float(ts), float(x)) )
            inside = False
            #et = int(ts / 1000000)
            et = int(ts * 1000)
            #eaf.add_annotation(sensor, st, et, value='Move')
            empty_time = et - previous_annotation[1] # not used
            previous_annotation = current_annotation
            current_annotation += [et, empty_time]
            annotations.append( current_annotation )
            current_annotation = []
    # we might have lost the last one if it is "inside" until the end.
    print( annotations )
    for annotation in annotations:
        if annotation[1] - annotation[0] > 250:
            eaf.add_annotation(sensor, annotation[0], annotation[1], value='Move')
            
eaf.to_file("mocap_valentijn/beach_repr_2_pb.eaf", pretty=True)

'''
Merging, list with intervals [start, end]
Fínd if start within x milliseconds, take the "most left" one
Find the "most right" one, that is largest in that "group"

'''

sys.exit(9)
# Group plots

plot_group( group_LHand_M, df, title="Left Hand Sensors" )
plot_group( group_RHand_M, df, title="Right Hand Sensors" )

#plot_group( group_Body, df_dists, title="Body" )

plot_groups_lr( group_LArm, group_RArm, df_dists, title="Left and Right Arm" )

#plot_group( group_RArm+group_RHand_M, pd.concat([df, df_dists]), title="TEST" )

# Create a new dataframe with "distance moved across threshold" indicators.
# Determine threshold through statistical analysis?
df_dists_t = pd.DataFrame()
df_dists_t["Timestamp"] = df_dists["Timestamp"]

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

axes[0].vlines(df_dists["Timestamp"].values,
               0, 2*df_dists_t["x_LElbowOut_T"].values) # colour according to real value?

axes[0].scatter(
    df_dists["Timestamp"].values,
    df_dists_t["x_LElbowOut_T"].values,
    s=siz, c=col,
    #c=df_dists["x_LElbowOut"].values, cmap='viridis'
)

#x, y = axes[1].pcolormesh(
#    df_dists["Timestamp"].values,
#    1,
#    df_dists["x_LElbowOut"].values,
#    cmap='RdBu', vmin=0, vmax=10)

## Make it 2D? on the y axis all the signals/sensors, and a colour value for the x-value of sensor

#mp.imshow(df_dists["x_LElbowOut"].values, cmap='jet')
#mp.pcolormesh( [df_dists["x_LElbowOut"]], cmap='Greys', shading='gouraud')

my_cmap = mp.get_cmap("viridis")
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
axes[1].bar(
    df_dists["Timestamp"].values,
    df_dists_t["x_LElbowOut_T"].values,
    color=my_cmap( rescale(df_dists["x_LElbowOut"].values) )
    #s=siz, c=col,
    #c=df_dists["x_LElbowOut"].values, cmap='viridis'
)
    
'''
axes[1].set_title("x_RElbowOut")
axes[1].plot(
    df_dists["Timestamp"].values,
    df_dists["x_RElbowOut"].values
)
'''
axes[1].legend(loc="upper right")
fig.tight_layout()

#plot_group( ["x_LElbowOut", "x_RElbowOut"], df_dists, title="Left and Right Elbow" )

'''
# Another distances plot
fig, axes = mp.subplots(nrows=2, ncols=1, figsize=(12,6), sharex=True, sharey=True)
fig.suptitle( "Distances Right and Left Arms" )
for sensor in group_LArm:
    axes[0].plot(
        df_dists["Timestamp"].values,
        df_dists[sensor].values
)
for sensor in group_RArm:
    axes[1].plot(
        df_dists["Timestamp"].values,
        df_dists[sensor].values
)
fig.tight_layout()
'''

# ----------------------------

# Plot
'''
fig, axes = mp.subplots(nrows=2, ncols=1, figsize=(12,6), sharex=True, sharey=True)
# "two groups combined"
for sensor in group_RFingers_M:
    axes[0].plot(
        "Timestamp",
        sensor,
        data=df
    )
#axes[1].legend(loc="upper right")
box = axes[0].get_position()
#axes[0].set_position([box.x0, box.y0 + box.height * 0.12, box.width, box.height * 0.88])
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
for sensor in group_LFingers_M:
    axes[1].plot(
        "Timestamp",
        sensor,
        data=df
    )
#axes[1].legend(loc="upper right")
box = axes[1].get_position()
#axes[1].set_position([box.x0, box.y0 + box.height * 0.12, box.width, box.height * 0.88])
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
fig.tight_layout()
'''
plot_groups_combined_stacked(group_LFingers_M, group_RFingers_M, df,
                             title="Left and Right Fingers",
                             subtitles=["Left", "Right"])

# Use "np.condition" to determine hand/finger/arm movements? (1/0 columns)

plot_group_combined(group_LFingers_M, df, title="LFingers_M combined") 

mp.show()
'''
# For creating new column with multiple conditions
conditions = [
    (df['Base Column 1'] == 'A') & (df['Base Column 2'] == 'B'),
    (df['Base Column 3'] == 'C')]
choices = ['Conditional Value 1', 'Conditional Value 2']
df['New Column'] = np.select(conditions, choices, default='Conditional Value 1')

siz = np.where(df["x_LWristOut_vel_M_T"]<200,0,1)

conditions = [
    df['gender'].eq('male') & df['pet1'].eq(df['pet2']),
    df['gender'].eq('female') & df['pet1'].isin(['cat', 'dog'])
]
choices = [5,5]
df['points'] = np.select(conditions, choices, default=0)
print(df)
'''
