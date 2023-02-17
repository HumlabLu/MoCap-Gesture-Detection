# MoCap 

Files to post-process mocap data for event detection and EAF file generation. Work in progress, needs some clean up.

### `mocap_gen_dists.py`

Calculates distances travelled by the sensors from the 3D coordinates. 

Example:
```shell
python mocap_gen_dists.py -f mocap_valentijn/beach_repr_2b.tsv
```

The above example creates a `mocap_valentijn/beach_repr_2b_dists.tsv` file containing the 3D distances travelled between each X,Y,Z point of each sensor.

### `mocap_gen_dirs.py`

Determine the direction of movement from the 3D data. Work-in-progress.

Example:
```shell
python mocap_gen_dirs.py -f mocap_valentijn/beach_repr_2b.tsv
```

The above creates a file called `mocap_valentijn/beach_repr_2b_dirs.tsv` containing values :-).

### `mocap_eaf.py`

Takes a distance file (generated by `mocap_gen_dists.py`) and tries to determine motion events. Adds a tier for each sensor with motion annotations to an existing EAF file. Tiers/sensors are hard-coded at the moment.

Example:
```shell
python mocap_eaf.py -f mocap_valentijn/beach_repr_2b_velocity_M.tsv -d mocap_valentijn/beach_repr_2b_dists.tsv -e mocap_valentijn/beach_repr_2_pb.eaf 
python mocap_eaf.py -d mocap_valentijn/beach_repr_2b_dists.tsv -D mocap_valentijn/beach_repr_2b_dirs.tsv -E mocap_valentijn/mynew.eaf
python mocap_eaf.py -d mocap_valentijn/beach_repr_2b_dists.tsv -D mocap_valentijn/beach_repr_2b_dirs.tsv 
```

### `mocap_plot.py`

Plots sensor data.

Example:
```shell
python mocap_plot.py -f mocap_valentijn/beach_repr_2b_velocity_M.tsv -d mocap_valentijn/beach_repr_2b_dists.tsv -r mocap_valentijn/beach_repr_2b_dirs.tsv
```

### `mocap_vel_00.py`

Plots velocity file.

### `mocap_acc.py`

Plots one of the angular acceleration files (LH).
