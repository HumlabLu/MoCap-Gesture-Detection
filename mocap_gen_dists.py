import pandas as pd
import math
import argparse
import os
import sys

# Use PYVENV in Development

# ============================================================================
# Creates a file with "distances" travelled by the sensors.
# Optionally resamples the data, saves each sensor stream in a
# separate file.
# The -w creates a 200 Hz 16bit int wave file which can be loaded
# into Elan for a visualisation of the movement.
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--filename", help="MoCap tsv file (3D positions).",
                     default="mocap_valentijn/beach_repr_2b.tsv" )
parser.add_argument( "-s", "--save", action="store_true",
                     help="Save individual sensors." )
parser.add_argument( "-w", "--wave", action="store_true",
                     help="Save each sensor as a pseudo wave file." )
args = parser.parse_args()

# ============================================================================
# Read the data.
# ============================================================================

# Data is index plus timestamp plus 64*3 data points?
# Read the original data, save in df_rows which is used later to calculate
# the distances.
df      = None
df_rows = []
lnum    = 0
freq    = 200 # Parse from file header.
with open(args.filename, "r") as f:
    for line in f:
        bits = line.split()
        if bits[0] == "FREQUENCY":
            freq = int(bits[1])
        if bits[0] == "MARKER_NAMES":
            column_names = bits[1:] # We add a new "Timestamp" name later.
            print( column_names )
        if len(bits) > 65:
            try:
                bits     = [ float(x) for x in bits ]
                triplets = [bits[i:i + 3] for i in range(2,len(bits)-2, 3)]
                df_rows.append( bits[1:] ) #skip index number
            except ValueError:
                print( "Skipping line", lnum )
        lnum += 1

# ============================================================================
# Create the distance data.
# ============================================================================

# Calculate the distance between twp triplets.
def dist3d(v0, v1):
    dist = sum( [ (x-y)*(x-y) for x,y in zip(v0, v1) ] )
    return math.sqrt( dist )

df_distances  = []
df_dists_rows = [ [0.0] * len(column_names) ] # init with zeros for timestamp 000000
row           = df_rows[0]
prev_triplets = [ row[i:i + 3] for i in range(1,len(row)-1, 3) ]
for ln, row in enumerate(df_rows[1:]):
    ts       = row[0] # timestamp
    new_row  = [ ts ]
    triplets = [ row[i:i + 3] for i in range(1,len(row)-1, 3) ]
    ti       = 0 # triplet index
    for t0,t1 in zip(triplets, prev_triplets):
        dist = dist3d( t0, t1 )
        if dist == 0:
            print( "Zero distance in line", ln, "at", ts, column_names[ti])
        if dist >100: # What unit?
            print( "Large distance", dist, "in line", ln, "at", ts, column_names[ti])
        new_row.append( dist )
        ti += 1
    prev_triplets = triplets
    df_dists_rows.append( new_row )

# Distances dataframe, use original column names.
column_names = ["Timestamp"] + column_names
df_dists     = pd.DataFrame(
    df_dists_rows,
    columns=column_names
)
print( df_dists.head() )
print( df_dists.tail() )

# ============================================================================
# Save the new dataframe.
# ============================================================================

filepath_bits  =  os.path.split( args.filename )
dists_filename = os.path.join( filepath_bits[0], filepath_bits[1][:-4] + "_dists.tsv" ) # Note, no error checking
print( filepath_bits, dists_filename )

# Save it into a new file.
df_dists.to_csv(
    dists_filename,
    index=False,
    sep="\t"
)
print( "Saved:", dists_filename )

# ============================================================================
# Save wave files and individual columns.
# ============================================================================

if args.save or args.wave:
    import scipy.io.wavfile as wv
    import numpy as np
    filepath_bits  =  os.path.split( dists_filename ) # Now we use dists filename!
    for col in column_names[1:]:
        data = df_dists[col]
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
            try:
                wv.write(wav_filename, freq, data.astype(np.int16)) #astype(np.float32))
            except pd.errors.IntCastingNaNError:
                print( "Error saving", wav_filename )
