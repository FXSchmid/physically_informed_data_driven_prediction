Trainer (train_multi_balanced) — Quick README


What to pass to the constructor
------------------------------
- `data_path` (str) — path to the folder that contains `processed_data`.
- `channels` (optional list) — channel sizes for the model (default: [32,64,128,256,512]).

Expected files (inside `<data_path>/processed_data`)
--------------------------------------------------
- `areas_train_multi.txt`
- `event_timestep_selected.txt`
- `event_timestep_hdf.txt`
- `Dataset.h5` (must include datasets used by the code: `x_gis`, `x_rain`, `x_water`, `y`, `x_water_assumed`)

