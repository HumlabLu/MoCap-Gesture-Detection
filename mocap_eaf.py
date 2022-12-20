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
# python mocap_eaf.py -d mocap_valentijn/beach_repr_2b_dists.tsv -e mocap_valentijn/beach_repr_2_pb.eaf

# Create/add to an existing EAF file.

# ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument( "-d", "--distsfilename",
                     help="MoCap tsv file (distances, from mocap_gen_dists.py)." )
parser.add_argument( "-D", "--dirsfilename",
                     help="MoCap tsv file (directions, from mocap_gen_dirs.py)." )
parser.add_argument( "-e", "--eaffilename",
                     help="EAF file to augment." )
parser.add_argument( "-E", "--eafoutfilename",
                     help="EAF file output." )
parser.add_argument( "-m", "--minimumlength", default=250, type=int,
                     help="Minimum annotation time in ms. Shorter will be ignored." )
parser.add_argument( "-g", "--minimumgap", default=120, type=int,
                     help="Minimum annotation gap in ms. Shorter will be merged" )
parser.add_argument( "-t", "--threshold", default=0.1, type=float,
                     help="Movement threshold, only consider larger than threshold*max" )
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

# Read the distance data and EAF file.

if not args.distsfilename:
    print( "No dists filename specified, quitting." )
    sys.exit(1)
df_dists = pd.read_csv(
    args.distsfilename,
    sep="\t"
)

if not args.eaffilename:
    print( "No EAF filename specified, creating a new EAF file." )
    eaf = pympi.Elan.Eaf(author='mocap_eaf.py')
else:
    eaf = pympi.Elan.Eaf(file_path=args.eaffilename, author='mocap_eaf.py')

if not args.eafoutfilename:
    if not args.eaffilename:
        print( "Need an EAF output filename (-E ...)." )
        sys.exit(1)
    else:
        args.eafoutfilename = args.eaffilename
        
# ----------------------------

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

for sensor in ["x_LHandIn", "x_LHandOut", "x_LWristIn", "x_LWristOut"]+["x_RHandIn", "x_RHandOut", "x_RWristIn", "x_RWristOut"]+["LATotal", "RATotal"]:
#for sensor in ["x_LHandIn", "x_LHandOut", "x_LWristIn", "x_LWristOut",
#               "x_RHandIn", "x_RHandOut", "x_RWristIn", "x_RWristOut",
#               "x_WaistLBack", "x_WaistRBack"]:# group_LArm+group_RArm:
    dist_max = df_dists[sensor].max()
    dist_min = df_dists[sensor].min()
    print()
    print( sensor, dist_min, dist_max )
    eaf.add_tier( sensor, ling='default-lt' )
    # instead of threshhold, difference in direction, we have that data?
    threshold = dist_min + (dist_max * args.threshold) # take if > "10%"
    inside = False
    st = -1
    et = -1
    annotations = []
    previous_annotation = [0, 0]
    current_annotation = [0, 0]
    for ts, x in zip(df_dists["Timestamp"].values, df_dists[sensor].values):
    #for ts, x in zip(df_dists.index.values, df_dists[sensor].values): # timedeltas are microseconds
        if not inside and x > threshold:
            #print( "NEW {:.3f} {:.4f}".format(float(ts), float(x)) )
            inside = True
            #st = int(ts / 1000000) # start time
            st = int(ts * 1000) # start time
            empty_time = st - previous_annotation[1] # to see if close to previous
            if empty_time < args.minimumgap: #arbitrary... 120ms
                #print( "Short", previous_annotation )
                st = previous_annotation[0] # cheat, and put the previous start time
                annotations = annotations[:-1] # and remove previous annotation.
            # add to annotations here?
            current_annotation = [ st ] 
            # concat annotations if close to gether? postprocess?
        elif not inside:
            pass
        elif inside and x <= threshold:
            #print( "--- {:.3f} {:.4f}".format(float(ts), float(x)) )
            inside = False
            #et = int(ts / 1000000)
            et = int(ts * 1000)
            #eaf.add_annotation(sensor, st, et, value='Move')
            annotation_time = et - current_annotation[0]
            previous_annotation = current_annotation
            current_annotation += [et, annotation_time]
            annotations.append( current_annotation )
            current_annotation = []
    # we might have lost the last one if it is "inside" until the end.
    #print( annotations )
    for annotation in annotations:
        if annotation[1] - annotation[0] > args.minimumlength:
            print( annotation )
            eaf.add_annotation(sensor, annotation[0], annotation[1], value='Move')
            
#eaf.to_file("mocap_valentijn/beach_repr_2_pb.eaf", pretty=True)

'''
Merging, list with intervals [start, end]
Fínd if start within x milliseconds, take the "most left" one
Find the "most right" one, that is largest in that "group"
Or just merge the rows into one.
'''

if not args.dirsfilename:
    eaf.to_file(args.eafoutfilename, pretty=True)
    print( "Wrote", args.eafoutfilename )
    sys.exit(1)
    
df_dirs = pd.read_csv(
    args.dirsfilename,
    sep="\t"
)

# x_HeadL_X_dir, x_HeadL_Y_dir, x_HeadL_Z_dir, etc
dir_labels = { # "N"egative, "P"ositive values
    'X': {"N":"Right",  "P":"Left"},
    'Y': {"N":"Foward", "P":"Backward"},
    'Z': {"N":"Down",   "P":"Up"}
}   
def label(sensor, val):
    axis = "X"
    if "_Y_" in sensor:
        axis = "Y"
    elif "_Z_" in sensor:
        axis = "Z"
    if val < 0:
        return dir_labels[axis]["N"]
    elif val > 0:
        return dir_labels[axis]["P"]
    else:
        return None

# We can use the distance groups, as it is the same data
for group in ["x_RHandOut"]: #group_LArm+group_RArm: #group_Head:
    for sensor in [group+"_Y_dir"]:#[group+"_X_dir", group+"_Y_dir", group+"_Z_dir"]:
        dist_max = df_dirs[sensor].max()
        dist_min = df_dirs[sensor].min()
        print()
        print( sensor, dist_min, dist_max )
        eaf.add_tier( sensor, ling='default-lt' )
        # instead of threshhold, difference in direction, we have that data?
        threshold = dist_min + (dist_max * args.threshold) # take if > "10%"
        inside = False
        st = -1
        et = -1
        annotations = []
        previous_annotation = [0, 0]
        current_annotation = [0, 0]
        sign = 0
        prev_sign = 0
        current_dir = 0
        for ts, x in zip(df_dirs["Timestamp"].values, df_dirs[sensor].values):
        #for ts, x in zip(df_dirs.index.values, df_dirs[sensor].values): # timedeltas are microseconds
            if x != 0:
                sign = math.copysign(1, x)
            if not inside and x != 0: # New annotation
                inside = True
                current_dir = label(sensor, sign) # The direction label ("up", etc)
                print( "NEW {} {:.4f} {}".format(int(ts*1000), float(x), current_dir) )
                #st = int(ts / 1000000) # start time
                st = int(ts * 1000) # start time
                empty_time = st - previous_annotation[1] # to see if close to previous
                if sign == prev_sign and empty_time < args.minimumgap: #arbitrary...
                    print( "Short, merge with", previous_annotation )
                    st = previous_annotation[0] # cheat, and put the previous start time
                    annotations = annotations[:-1] # and remove previous annotation.
                # add to annotations here?
                current_annotation = [ st ]
                # concat annotations if close to gether? postprocess?
            elif not inside:
                pass
            elif inside and x!=0 and sign != prev_sign:
                print( "Flip" )
                # reversal is also an end/start but we stay "inside"
                et = int(ts * 1000) # end time of the current, ending now
                st = int(ts * 1000) # start time of the new one, starting now
                annotation_time = et - current_annotation[0]
                previous_annotation = current_annotation
                current_annotation += [et, annotation_time, label(sensor, prev_sign)]
                annotations.append( current_annotation )
                current_annotation = [ st ] # new annotation starting here
            elif inside and x==0:
                #print( "END {:.3f} {:.4f}".format(float(ts), float(x)) )
                inside = False
                #et = int(ts / 1000000)
                et = int(ts * 1000)
                #eaf.add_annotation(sensor, st, et, value='Move')
                annotation_time = et - current_annotation[0]
                previous_annotation = current_annotation
                current_annotation += [et, annotation_time, label(sensor, sign)]
                print( "END", current_annotation )
                annotations.append( current_annotation )
                current_annotation = []
            if sign != 0:
                prev_sign = sign # we need to detect "sign flip"
        # we might have lost the last one if it is "inside" until the end.
        #print( annotations )
        for annotation in annotations:
            if True or annotation[1] - annotation[0] > args.minimumlength:
                print( annotation )
                eaf.add_annotation(sensor, annotation[0], annotation[1], value=annotation[3])

eaf.to_file(args.eafoutfilename, pretty=True)
print( "Wrote", args.eafoutfilename )
