import pandas as pd
import math
import sys, re, os
import numpy as np
import datetime
from decimal import *

# ============================================================================
# Reads a TSV file.
# ============================================================================

'''
NO_OF_FRAMES	28613
NO_OF_CAMERAS	20
NO_OF_MARKERS	64
FREQUENCY	200
NO_OF_ANALOG	0
ANALOG_FREQUENCY	0
DESCRIPTION	--
TIME_STAMP	2022-06-02, 15:49:44.620	25975.44751328
DATA_INCLUDED	3D
MARKER_NAMES	x_HeadL	x_HeadTop	x_HeadR	x_HeadFront	...
TRAJECTORY_TYPES	Measured	Measured	Measured ...
150.884	3.315 ...
'''

class MoCapReader():
    def __init__(self):
        self.df = None
        self.df_dist = None
        self.df_vel = None
        self.df_acc = None
        self.info = {}

    def get_df(self):
        return self.df

    def get_df_dist(self):
        return self.df_dist

    def get_df_vel(self):
        return self.df_vel

    def get_df_acc(self):
        return self.df_acc

    def get_info(self):
        return self.info

    def is_stringline(self, thing):
        return re.match(r'[A-Z_]+', thing)

    def dist3d(self, v0, v1):
        assert( len(v0) == len(v1) )
        dist = sum( [ (x-y)*(x-y) for x,y in zip(v0, v1) ] )
        return math.sqrt( dist )

    def read(self, filename):
        self.filename = filename
        self.info = {}
        self.df = None
        self.df_dist = None
        self.num_columns = 0
        self.column_names = []
        self.num_new_columns = 0
        self.new_column_names = []
        df_rows = []
        df_dist_rows = []
        df_times = []
        lnum = 0
        freq = 200 # Default, parse this from file header.
        frame_counter = 0
        time_index    = 0
        tripets = []
        prev_triplets = []
        skip_start = False
        num_markers = 0
        with open(self.filename, "r") as f:
            for line in f:
                bits = line.split()
                if len(bits) == 0:
                    continue
                bit0 = bits[0]
                if self.is_stringline(bit0): 
                    if len(bits) == 2:
                        self.info[bit0] = bits[1]
                        if bit0 == "FREQUENCY":
                            freq = int(bits[1])
                    else:
                        self.info[bit0] = " ".join(bits[1:])
                    if bit0 == "MARKER_NAMES":
                        self.column_names = bits[1:]
                        if self.column_names[-1] == "start":
                            skip_start = True
                            self.column_names = self.column_names[:-1]
                            num_markers -= 1
                        self.num_columns = len(self.column_names)
                        self.num_new_columns = self.num_columns * 3
                        print( "num_markers", num_markers, len(self.column_names), self.num_new_columns )
                    if bit0 == "NO_OF_MARKERS":
                        num_markers = int(bits[1]) # Causes problem if false
                else: # Not a 'str', expect a line with only numeric data.
                    try:
                        bits = [ float(x) for x in bits ]
                        if skip_start:
                            bits = bits[:-3]
                        num_bits = len(bits)
                        assert( num_bits // 3 == num_markers )
                        if self.num_new_columns == num_bits:
                            # In this case we miss two colums, the frame number and timestamp.
                            # We generate them instead. Some tsv files appear to be mising them. 
                            times = [ frame_counter, time_index ]
                            frame_counter += 1
                            time_index += 1.0 / freq
                        else:
                            times = bits[0:2]
                            bits = bits[2:]
                        triplets = [bits[i:i + 3] for i in range(0, num_markers*3, 3)]
                        '''
                        if skip_start:
                            #triplets = [bits[i:i + 3] for i in range(0, num_bits-3, 3)] # was num_bits-3
                            triplets = [bits[i:i + 3] for i in range(0, num_markers*3, 3)] # was num_bits-3
                        else:
                            #triplets = [bits[i:i + 3] for i in range(0, num_bits, 3)]
                            triplets = [bits[i:i + 3] for i in range(0, num_markers*3, 3)]
                        '''
                        dists = []
                        for t0,t1 in zip(triplets, prev_triplets):
                            dist = self.dist3d( t0, t1 )
                            dists.append( dist )
                        prev_triplets = triplets
                        if len(dists) == 0:
                            dists = [0.0] * len(triplets)
                        df_dist_rows.append( dists )
                        df_times.append( times )
                        df_rows.append( bits )
                    except ValueError:
                        print( "Skipping line", lnum )
                lnum += 1
        self.new_column_names = [] 
        self.times_column_names = ["FrameCounter", "Timestamp"]
        for col in self.column_names:
            # We have three values for each sensor.
            for coord in ["_X", "_Y", "_Z"]: 
                self.new_column_names.append( col+coord )
        self.df = pd.DataFrame(
            np.concatenate( (df_times, df_rows), axis=1),
            columns=self.times_column_names + self.new_column_names
        )
        self.df = self.df.drop(["FrameCounter"], axis=1) # We don't use this one anyway
        self.df['td'] = pd.to_timedelta(self.df['Timestamp'], 's') # Create a Timestamp column from td.
        self.df = self.df.set_index(self.df['td']) # and use it as index
        self.timestamp = self.df['Timestamp'] # save it
        self.df = self.df.drop(["Timestamp"], axis=1) # Remove them, we have
        self.df = self.df.drop(["td"], axis=1) # the index now.
        print( self.df )
        #
        self.dist_column_names = ["FrameCounter", "Timestamp"]
        for col in self.column_names:
            self.dist_column_names.append( col+"_d3D" )
        self.df_dist = pd.DataFrame(
            np.concatenate( (df_times, df_dist_rows), axis=1),
            columns=self.dist_column_names
        )
        self.df_dist = self.df_dist.drop(["FrameCounter"], axis=1) # We don't use this one anyway
        self.df_dist['td'] = pd.to_timedelta(self.df_dist['Timestamp'], 's') # Create a Timestamp column from td.
        self.df_dist = self.df_dist.set_index(self.df_dist['td']) # and use it as index
        self.df_dist = self.df_dist.drop(["Timestamp"], axis=1) # Remove them, we have
        self.df_dist = self.df_dist.drop(["td"], axis=1) # the index now.
        print( self.df_dist )
        #
        # We can convert it to velocities?
        self.df_vel = self.df_dist.diff().fillna(0)
        self.df_vel = self.df_vel * freq # Maybe not, keep it frame based?
        self.dist_column_names = []
        for col in self.column_names:
            self.dist_column_names.append( col+"_vel" )
        self.df_vel.columns = self.dist_column_names
        print( self.df_vel )
        #
        # And acceleration
        self.df_acc = self.df_vel.diff().fillna(0)
        self.df_acc = self.df_acc * freq 
        self.dist_column_names = []
        for col in self.column_names:
            self.dist_column_names.append( col+"_acc" )
        self.df_acc.columns = self.dist_column_names
        print( self.df_acc )
        #
        return True

    # ============================================================================
    # Resample the data, in place.
    # The resample argument should be a string like "50ms".
    # ============================================================================
    def resample(self, resample):
        self.df = self.df.resample(resample).sum() # or mean()
        
    # ============================================================================
    # Should be done elsewhere?
    # ============================================================================
    def filter(self): 
        '''
        filtered_columns = ["Timestamp"]
        for sensor in df_pos.columns:
            for filter_re in args.filter:
                if re.search( filter_re, sensor ):
                    filtered_columns.append( sensor )
        if len(filtered_columns) == 1: # If only "Timestamp" then take all.
            filtered_columns = df_pos.columns
        df_pos = df_pos[filtered_columns]# Not necessary...
        '''
        pass

    # ============================================================================
    # Count the number of zero values.
    # Returns a dict with column name and (zero-count, frames).
    # ============================================================================
    def count_zero(self):
        res = {}
        for col in self.df.columns:
            count = (self.df[col] == 0).sum()
            frames = len(self.df[col])
            pct = count * 100.0 / frames 
            #print( f"{col:<20} {count:8d} ({pct:5.1f}%)" )
            res[col] = (count, frames)
        return res

    # ============================================================================
    # Resave the file without the header. Adds an "_r" suffix to the
    # filename. The timedelta index is temporarily converted to seconds.
    # ============================================================================
    def save(self, index=True):
        filepath_bits = os.path.split( self.filename )
        new_filename = os.path.join(filepath_bits[0], filepath_bits[1][:-4] + "_r.tsv" )
        current_index = self.df.index
        self.df.index = self.df.index.total_seconds() 
        self.df.to_csv( new_filename, sep='\t', index=index )
        self.df.index = current_index

'''
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
'''

if __name__ == "__main__":
    r = MoCapReader()
    for fn in [ "dyad_description_1_light_ev.tsv",
                "dyad_description_eng_1.tsv",
                "farm_adapt_2.tsv",
                "street_adapt_1.tsv"]:
        print( "=" * 80 )
        print( fn )
        print( "=" * 80 )
        res = r.read( fn )
        if not res:
            print( "Error reading", fn )
            sys.exit(1)
        
