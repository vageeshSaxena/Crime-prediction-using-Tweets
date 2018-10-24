# Crime Prediction with Tweets in Chicago

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/kstrauch94/Crime-Prediction-with-Tweets/master?filepath=Pipeline.ipynb)

## Preprocessing
The preprocessing of crimes incidents and tweets has already been done, and its
result are saved in this repo. In case you would like to preform it by yourself
again (takes ~ 2 hours), follow these steps:
1. Make sure that the directory `data/processed/` exists and empty.
2. Place all the JSON files from [Raw Collected
Tweets](https://www.dropbox.com/sh/uziw9ux45miwj6j/AACaO-sLWnRLah5gwiXEZTRpa?dl=0) in `data/raw/tweets/`.
3. Export [Chicago Crimes - 2001 to
present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2)) as CSV file, and place it in `data/raw/`
4. Run the Pipeline Jupyter Notebook.
