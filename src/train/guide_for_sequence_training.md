Training_sequence network — Usage Guide

This document explains how to use the Training_sequ class in
`src/train/train_sequence.py`. It covers the data layout, minimal examples,
and common troubleshooting steps.

1) Quick summary
----------------
- Class: `Training_sequ` (in `src/train/train_sequence.py`)
- Constructor now takes one required parameter: `data_path` (string).
- Optional: `channels` list to override model channel sizes.
- Typical usage: instantiate with `Training_sequ(data_path=<path>)` and call `.train_sequ()`.


2) Data layout expected
-----------------------
Given `data_path`, the code expects a `processed_data` folder inside it, with the following files (names used by the current code):

- `areas_train_multi.txt` — integers used for area indices
- `event_rotation.txt` — integers used for event rotations
- `rotation_area_hdf.txt` — table mapping (area, rotation) -> HDF row index
- `Dataset.h5` — HDF5 file that must contain datasets named at least:
  - `x_gis` (N, 256, 256, 8)
  - `x_rain` (N, 16, 16, 22)
  - `x_water` (indexes/slices used to build input water stacks)
  - `y` (N, 128, 128, 11)
  - `x_water_assumed` (N, 256, 256, 1)

Notes:
- If your HDF file uses a different file name, either rename the file to `Dataset.h5` or update the filename constant in `train_sequence.py`.
- Shapes listed above are those the code currently allocates; if your data uses different shapes, you must adapt the code or reformat your HDF datasets.

3) Hyperparameter 
-----------------------------
All hyperparemeters have to be set in the config.py class

4) Loading pretrained weights
-----------------------------
If you have pretrained weights, set the attribute and call `prepare_network`:

```python
trainer.weights_path = r"D:\path\to\weights.hdf5"
trainer.prepare_network()
```

`prepare_network` will try to build the model from example tensors if available, and will attempt loading only if `weights_path` points to an existing file.


