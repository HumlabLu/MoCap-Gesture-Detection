# MoCap 

Files to post-process mocap data for event detection and EAF file generation.

### `mocap_gen_dists.py`

Calculates distances travelled by the sensors from the 3D coordinates.

### `mocap_gen_dirs.py`

Determine the direction of movement from the 3D data.

### `mocap_04.py`

Takes a distance file (generated by `mocap_gen_dists.py`) and tries to determine motion events. Adds a tier for each sensor with motion annotations to the EAF file.




