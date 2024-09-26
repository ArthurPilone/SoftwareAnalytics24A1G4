# Data Collection
### How we collected the data for our model
## SA Group 4

## Issue Data


## Commits Data

Firstly, we cloned the VSCode git repository on Sept 26th, at 11:20 AM.
```
git clone git@github.com:microsoft/vscode.git
```

Then, we extracted the data relative to the changes in every commit in the repositoriy:
```
git log --name-status --date=iso --pretty=format:"<commit-id>%h</commit-id><author-login>%al</author-login><date>%ad</date><message>%s</message><body>%b</body>"> ../log.txt
```

Data such as commit date or files changed could be used in the future to train the model, but we're mainly interested in the '\<author-login>' field of every commit.

Using this data, we can later match the information collected alongside the issues, to the number of commits by each contributor. This is done using the `log.txt` file (not commited to the repo) and the `contribution_counter.ipynb` notebook. 


