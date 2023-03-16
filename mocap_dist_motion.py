import pandas as pd
import math
import argparse
import os
import sys
from MoCap.File import MoCapReader

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
parser.add_argument( "-s", "--save", action="store_true",
                     help="Save individual sensors." )
parser.add_argument( "-o", "--outfile", default=None, type=str,
                     help="Save motion data in outfile." )
args = parser.parse_args()

if not args.outfile:
    param_str = f"t{args.threshold}l{args.minlength}g{args.mingap}"
    filepath_bits  = os.path.split( args.filename )
    motion_filename = os.path.join( filepath_bits[0], filepath_bits[1][:-4] + "_dm_"+param_str+".csv" )
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

df_dists = r.get_df_dist()
#df_dists = r.get_df_vel()
print( df_dists )

for cn, col in enumerate(df_dists):
    col_max = df_dists[col].max()
    col_min = df_dists[col].min()
    col_threshold = (col_max-col_min)*(args.threshold/100.0) + col_min
    count = (df_dists[col] > col_threshold).sum()
    #print( cn, col, col_min, col_max, col_threshold, count )
    # add a columns if larger than xxx
    values = df_dists[col].values
    times = df_dists.index.total_seconds()
    in_anno = False
    anno_start = 0
    anno_end = 0
    anno_frames = 0
    annotations = []
    for t, v in zip(times, values):
        #print( t,v )
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
            anno_end = t
            annotations.append( (anno_start, anno_end, (anno_end-anno_start), col) )
            in_anno = False
    if in_anno:
        anno_end = t
        annotations.append( (anno_start, anno_end, (anno_end-anno_start), col) )
        in_anno = False
    with open(motion_filename, "a") as f:
        col_name = col[:-4] # Remove the _d3D bit at the end
        print( col_name )
        f.write( col_name+"\n" )
        for anno in annotations:
            if anno[2] > args.minlength/1000.0: # Print is longer than 0.5 seconds
                print( f"{anno[0]:6.3f}-{anno[1]:6.3f} {anno[2]:.3f}" )
                f.write( f"{anno[0]}, {anno[1]}, {round(anno[2],3)}\n" )
        print()
        f.write( "\n" )
print( "Saved in", motion_filename )
