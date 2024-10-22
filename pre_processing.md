# Pre-Processing Documentation

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