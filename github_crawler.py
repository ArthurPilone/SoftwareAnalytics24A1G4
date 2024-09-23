"""
	Module containing the GitHub issue crawler

	NOTE that this code requires a personal authentication token from GitHub.
	You must create or access a token as indicated on
	https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
	and put its value in the corresponding key value () int the .env file.

"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(os.path.abspath('')).absolute()))

import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from github import Github, Auth, BadCredentialsException

from issue import Issue

load_dotenv()

ISSUES_PER_PAGE = 100
REQUEST_TIMEOUT = 120  #in seconds

#pylint: disable=duplicate-code


def parse_args():
	'''
   	    Parses command line arguments
   	'''
	arg_parser = argparse.ArgumentParser()

	arg_parser.add_argument("-n",
	                        "--name",
	                        action='store',
	                        required=True,
	                        help='Project Name')
	arg_parser.add_argument("-p",
	                        "--project",
	                        action='store',
	                        required=True,
	                        help='Project Namespace in GitHub [group/project]')
	arg_parser.add_argument("-o",
	                        "--output",
	                        action='store',
	                        required=True,
	                        help='Path to output file')
	arg_parser.add_argument("-l",
	                        "--label",
	                        action='store',
	                        required=False,
	                        help='Only collect issues with the given label')

	args = arg_parser.parse_args()

	return args


class GitHubCrawler():  # pylint: disable=too-few-public-methods
	"""
		Scraper issues from a GitHub repository
	"""

	def __init__(  #pylint: disable=too-many-arguments
	        self, project_origin: str, persistency_path: str,
	        repository_namespace: str, label: str):
		self.project_origin = project_origin
		self.persistency_path = persistency_path
		self.repository_namespace = repository_namespace
		self.label = label
		super().__init__()

	def collect_issues(self):
		"""
			Returns an issue repository with the issues scraped
		"""

		print("Connecting to GitHub")
		# using an access token
		auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

		# Public Web Github
		github_connection = Github(auth=auth)

		try:
			repo = github_connection.get_repo(self.repository_namespace)
		except BadCredentialsException:
			print("-" * 80)
			print("ERROR: Invalid GitHub access token.")
			print(
			    "Make sure you have inserted a valid oauth token in the .env file"
			)
			sys.exit(1)

		print("Collecting Issues from the project")
		if self.label is not None:
			collected_issues = repo.get_issues(state="all", labels=[self.label])
		else:
			collected_issues = repo.get_issues(state="all")

		print(collected_issues[0].assignee)

		issues_as_dicts = []
		print("Parsing collected issues.\nThis might take a few minutes")

		for issue in tqdm(collected_issues[:1000]):
			# print(vars(issue))

			new_issue = Issue(str(issue.number),
							  issue.assignee,
			                  issue.title,
							  body=issue.body,
			                  creation_time=issue.created_at,
			                  completion_time=issue.closed_at)
			
			issues_as_dicts.append(new_issue.to_dict())

		return pd.DataFrame.from_dict(issues_as_dicts)


def main():
	"""
	Main code for executing the crawler script
	"""
	args = parse_args()

	collector = GitHubCrawler(args.name, args.output, args.project
	                          , args.label)

	resulting_issues = collector.collect_issues()

	print(len(resulting_issues), "issues collected")

	resulting_issues.to_csv(args.output, compression='gzip', index=False)


if __name__ == "__main__":
	main()
