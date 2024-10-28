
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

## Pre-Processing Documentation

## File Path

The dataset is initially loaded from a compressed CSV file:

```
DATA_FILE_PATH = "vscode_no_prs.csv.gzip
sample_dataset = pd.read_csv(DATA_FILE_PATH, compression='gzip', lineterminator='\n')
```

The dataset is read into a list of dictionaries using to_dict('records') and then processed.

## Number of times as an Assignee

During data loading, the number of times each author is assigned as an Assignee is recorded in a dictionary, which will later be used to filter out authors who have been assigned as an Assignee too few times (less than or equal to once).

## Pre-Processing Functions

## 1. Filtering Functions

Several filtering functions are applied to the dataset to select the appropriate issues for training and testing.

### filter_test_dataset

This function selects issues with an identifier in the test range (210000 < id <= 220000). If the issue falls outside this range, it is excluded.

```
def filter_test_dataset(issue):
    issue_id = int(issue.identifier)
    if 210000 < issue_id <= 220000:
        return issue
    return None
```

### filter_main_training_dataset

This function selects issues for the main training dataset, filtering out any issues with an identifier greater than 210000.

```
def filter_main_training_dataset(issue):
    issue_id = int(issue.identifier)
    if issue_id <= 210000:
        return issue
    return None
```

### filter_recent_issues_training_dataset

This function filters issues based on a more recent subset of the dataset with identifiers in the range 190000 <= id <= 210000.

```
def filter_recent_issues_training_dataset(issue):
    issue_id = int(issue.identifier)
    if 190000 <= issue_id <= 210000:
        return issue
    return None
```

### filter_basic_trainingset_requirements

This function ensures that only issues that are closed and have exactly one assignee are included in the dataset.

```
def filter_basic_trainingset_requirements(issue):
    if issue.completion_time is None or issue.assignee is None or (isinstance(issue.assignee, list) and len(issue.assignee) != 1):
        return None
    return issue
```

### filter_unfrequent_commiters

This function filters out issues from authors with commit counts below a specified threshold. The default threshold is 1.

```
def filter_unfrequent_commiters(issue):
    threshold = 1
    author = issue.assignee
    if author is None or commit_no_by_author.get(author, 0) <= threshold:
        return None
    return issue
```

## 2. Cleaning Functions

These functions clean up the issue fields, such as title and body, by removing unnecessary content.

### clean_issue_title

This function cleans the summary field of the issue by removing unnecessary elements like mentions of other issues, monospacing markdown, etc.

```
def clean_issue_title(issue):
    new_title = issue.summary
    new_title = re.sub(r"\[?\s*[Ff]ollow up to #?[\d]+\s*\]?", "", new_title)
    new_title = re.sub(r"`([\s\S]*?)`", r"\1", new_title)
    issue.summary = new_title
    return issue
```

### clean_issue_body

This function cleans the body field of the issue, surrounding code fragments with sentinel tokens (<BoC> and <EoC>) and removing unnecessary content like headers, markdown formatting, and HTML tags.

```
def clean_issue_body(issue):
    issue_body = issue.body
    if issue_body is None:
        return ""
        
    code_fragments = ""
    new_body = issue_body
    for match in re.findall(r"```([\s\S]*?)```", new_body):
        code_fragments += CODE_BEGIN_SENTINEL + match + CODE_END_SENTINEL + "\n"
    new_body = re.sub(r"```([\s\S]*?)```", "", new_body)
    new_body = re.sub(r"<[\s\S]*?>", r"", new_body)
    new_body = code_fragments + new_body
    issue.body = new_body
    return issue
```

## Dataset Processing Pipeline

The following pipeline is used to apply the filtering and cleaning functions:

```
clean_dataset = apply_steps_to_dataset([
    filter_basic_trainingset_requirements,
    filter_unfrequent_commiters,
    clean_issue_title,
    clean_issue_body
], issues)

main_training_dataset = apply_steps_to_dataset([filter_main_training_dataset], clean_dataset)
recent_issues_training_dataset = apply_steps_to_dataset([filter_recent_issues_training_dataset], clean_dataset)
test_dataset = apply_steps_to_dataset([filter_test_dataset], clean_dataset)
```

The apply_steps_to_dataset function iterates over each issue in the dataset and applies the provided list of pre_processing functions sequentially.

## Saving Processed Datasets

The processed datasets are saved as compressed CSV files for further use. The following function converts the list of issues to a CSV file:

```
def save_issue_repo(new_path, issue_repo):
    issues_as_dicts = [issue.to_dict() for issue in issue_repo]
    issues_as_dataset = pd.DataFrame.from_dict(issues_as_dicts)
    issues_as_dataset.to_csv(new_path, compression='gzip', index=False)
```

The processed datasets are saved to the following paths:

	•	train_A.csv.gzip: Main training set
	•	train_B.csv.gzip: Recent issues training set
	•	test.csv.gzip: Test set

This concludes the pre-processing steps for the dataset. Each step ensures that only the relevant, clean data is used for training and testing machine learning models.


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

