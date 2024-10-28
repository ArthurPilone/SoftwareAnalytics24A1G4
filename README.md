
## Installation


This project uses Python v3.10.4 and the [venv module](https://docs.python.org/3/library/venv.html). 


Before installing the dependencies for the first time, you will need to create a virtual environment. As the project depends on Python v3.10.4, make sure that this is the current python version used in the moment that the virtual environment is created.

## Setting up Python v3.10.4 using `pyenv`

If your current python installation is not 3.10.4, you may use a solution such as [pyenv](https://github.com/pyenv/pyenv). After following [pyenv's installation walkthrough](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation), you can install the desired Python version by doing:

```
$ pyenv install 3.10.4
``` 

Afterwards, switch the python version used in the current shell by doing:

```
$ pyenv shell 3.10.4
```

For problems with `pyenv` installation or usage, please refer to [its official documentation](https://github.com/pyenv/pyenv?tab=readme-ov-file).

## Creating a Virtual Environment and Installing Dependencies

We use a virtual environment in order to maintain, organize and share the dependencies of our system.

_Before creating a virtual environment, make sure that the Python version you're currently using is v.3.10.4. Run `python -V` and make sure the output is equivalent to `Python 3.10.4`_.

Create a virtual environment for the project doing:
```
$ python -m venv ./.venv
```

Before running the code for the first time, remember to install the required python dependencies using our newly created virtual environment. Firstly, activate the virtual environment using:
```
$ source .venv/bin/activate
```

Then, install the dependencies specified in `requirements.txt` by using `pip`

```
(.venv) $ pip install -r requirements.txt
```

## Executing

After creating a virtual environment with the required dependencies, one must always activate the virtual environment before executing the system's scripts. If you haven't activated the virtual environment in the current shell, do so:
```
$ source .venv/bin/activate
```
_Note that this might not be necessary if you've just finished installing the system's dependencies using the current shell._

## Pre-Processing

Run the notebook `pre_processing.ipynb` on the local machine.
Complete each cell that will produce next output files:

	•	train_A.csv.gzip: Main training set
	•	train_B.csv.gzip: Recent issues training set
	•	test.csv.gzip: Test set

This concludes the pre-processing steps for the dataset.

## Running the Model Training

Once the data is pre-processed, you can train the models using `train_test.py.` To enable GPU training, execute the script with the `"CUDA_VISIBLE_DEVICES=0` command:
```
$ CUDA_VISIBLE_DEVICES=0 python train_test.py
```

This script will train two models:

`main_issue_classifier`: Trained on train_A.csv.gzip (main dataset).
`secondary_issue_classifier`: Trained on train_B.csv.gzip (recent dataset).
Training logs will be saved in `main_training_log.txt` and `secondary_training_log.txt`, and checkpoints will be saved in `training_main_model` and `training_secondary_model`.

## Running the Prediction Script
The first thing before running the prediction script we have to count the number of commits by each contributor for Train set A and B. We are doing it with `contribution_counter.py` which will create 2 separate commit_counts csvs:
```
$ python contribution_counter.py
```
To assign an assignee to a specific issue, use `bug_triage_predictor.py`. This script will predict the assignee for a given issue ID from the test set and display the number of commits by the predicted assignee. 
```
$ CUDA_VISIBLE_DEVICES=0 python bug_triage_predictor.py
```
You’ll be prompted to enter an issue ID. The script will use both models to predict an assignee and show the number of completed issues by the assignee, using commit data collected during pre-processing.

