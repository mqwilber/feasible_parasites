# feasible_parasites

This repo contains the code necessary to reproduce the results found in `docs/feasible_manuscript.pdf` and `docs/supplementary_material.pdf`.  The data used for this analysis is not yet publicly distributable, so this repository **does not** contain the entire workflow from data to compiled manuscript. Rather, we have included a `data/archival/dummy_data.csv` file that allows the various scripts to be executed, but produces dummy results (because the dummy data is not real).  Therefore, do not be surprised when executing `run_all.sh` produces completely different results than what you see in `docs/*` (because it should!). A more detailed description of each folder in this repo is given below

`code`

- `agg_fxns.py`: This script contains the necessary functions for testing whether a given host-parasite distribution follows a top-down or bottom-up model.
- `test_agg_fxns.py`: The script provides unit tests for the functions in `agg_fxns.py`
- `manuscript_analysis_*.py`: Scripts that use `agg_fxns.py` to perform top-down and bottom-up analyses on parasite data. Each script contains a description of what it does at the beginning of the script.
- `manuscript_plots.*`: Either a Python script or IPython Notebook for making the figures include in `docs/*`
- `run_all.sh`: A bash script that shows the order in which the manuscript analysis scripts should be run. Executing this script will run the analysis on the dummy data. 

`docs`

- `feasible_manuscript.pdf`: A draft of the manuscript for which the scripts in `code` are used.
- `supplementary_material.pdf`: A draft of the supplementary material for which the scripts given in `code` are used.

`results`

  - Contains the various plots and pickled data generated from the `*.py` scripts described above using the dummy data

`data`

- `archival`
  - `dummy_data.csv`: A dummy data file so the the above scripts can be executed.
  - `make_dummy_data.py`: Python script for generating the dummy data
