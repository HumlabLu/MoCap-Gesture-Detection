import pandas as pd
import math
import argparse
import os
import sys
from MoCap.File import MoCapReader
from scipy.stats.mstats import winsorize

# Use PYVENV in Development

# ============================================================================
# Generate motion
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--filename", help="MoCap tsv file (3D positions).",
                     default="mocap_valentijn/beach_repr_2b.tsv" )
parser.add_argument( "-t", "--threshold", help="Threshold in percent.",
                     default=10, type=int )
parser.add_argument( "-m", "--minlength", help="Minimum length in milliseconds",
                     default=250, type=int )
parser.add_argument( "-g", "--mingap", help="Minimum gap in milliseconds",
                     default=100, type=int )
parser.add_argument( "-s", "--signal", default="dis",
                     help="Which signal to use (dis, vel, acc)" )
parser.add_argument( "-S", "--save", action="store_true",
                     help="Save individual sensors." )
parser.add_argument( "-o", "--outfile", default=None, type=str,
                     help="Save motion data in outfile." )
parser.add_argument( "-W", "--winsorise", default=0.0, type=float,
                     help="Clip data to N (eg 0.0001) percentiles." )
args = parser.parse_args()

if not args.outfile:
    if args.winsorise:
        w_str = str(args.winsorise)[2:]
        param_str = f"{args.signal}_t{args.threshold}l{args.minlength}g{args.mingap}W{w_str}"
    else:
        param_str = f"{args.signal}_t{args.threshold}l{args.minlength}g{args.mingap}W0"
    filepath_bits  = os.path.split( args.filename )
    motion_filename = os.path.join( filepath_bits[0], filepath_bits[1][:-4] + "_m_"+param_str+".csv" )
else:
    motion_filename = args.outfile
if os.path.exists( motion_filename ):
    print( "File exists", motion_filename )
    sys.exit(1)

# ============================================================================
# Read the data.
# ============================================================================

r = MoCapReader()
res = r.read( args.filename )
if not res:
    print( "Error reading", args.filename )
    sys.exit(1)

# ============================================================================
# Create the motion-from-distance data.
# ============================================================================

if args.signal == "vel":
    df_sig = r.get_df_vel()
elif args.signal == "acc":
    df_sig = r.get_df_acc()
else:
    df_sig = r.get_df_dist()
print( df_sig )

for cn, col in enumerate(df_sig):
    if args.winsorise > 0.0:
        percentiles = [args.winsorise, args.winsorise] # take away extreme values.
        df_sig[col] = winsorize(df_sig[col], percentiles)
    col_max = df_sig[col].abs().max()
    col_min = df_sig[col].abs().min()
    col_threshold = (col_max-col_min)*(args.threshold/100.0) + col_min
    count = (df_sig[col].abs() > col_threshold).sum()
    #print( cn, col, col_min, col_max, col_threshold, count )
    # add a columns if larger than xxx
    values = df_sig[col].abs().values
    times = df_sig.index.total_seconds()
    in_anno = False
    anno_start = 0
    anno_end = 0
    anno_frames = 0
    num_frames = 0
    annotations = []
    for t, v in zip(times, values):
        #Print( t,v )
        num_frames += 1
        if not in_anno and v > col_threshold: # Potential new annotation.
            gap = t - anno_end
            if gap < args.mingap/1000.0: # Less than minimum gap, continue the "motion"
                in_anno = True
                anno_end = t
                annotations = annotations[:-1] # Remove last one, we continue.
                # adjust number of frames?
                continue
            anno_start = t
            anno_end = t
            anno_frames = 1
            in_anno = True
            continue
        if in_anno and v > col_threshold:
            anno_end = t 
            anno_frames += 1
            continue
        if in_anno and v <= col_threshold:
            anno_end = t #- r.get_period()
            annotations.append( (anno_start, anno_end, (anno_end-anno_start), col) )
            in_anno = False
    if in_anno:
        anno_end = t #- r.get_period()
        annotations.append( (anno_start, anno_end, (anno_end-anno_start), col) )
        in_anno = False
    # ---
    with open(motion_filename, "a") as f:
        col_name = col[:-4] # Remove the _d3D bit at the end
        print( col_name )
        f.write( col_name+"\n" )
        last_end = 0
        markers = []
        for anno in annotations:
            if anno[2] > args.minlength/1000.0: # Print is longer than 0.5 seconds
                gap = anno[0] - last_end
                if gap > 0:
                    print( f"{gap:7.3f} gap" )
                    markers +=  [0] * int(gap/r.get_period()) 
                print( f"{anno[0]:7.3f}-{anno[1]:7.3f} {anno[2]:.3f}" )
                f.write( f"{anno[0]}, {anno[1]}, {round(anno[2],3)}\n" )
                markers +=  [1] * int(anno[2]/r.get_period()) 
                last_end = anno[1]
        # Fill markers to end.
        markers += [0] * (num_frames - len(markers)) 
        print()
        #for i,(a,m) in enumerate(zip(values,markers)):
        #    print( i/r.get_freq(), a, m )
        f.write( "\n" )
print( "Saved in", motion_filename )
