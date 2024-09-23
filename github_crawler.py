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
from tqdm import tqdm
from dotenv import load_dotenv
from github import Github, Auth, BadCredentialsException

from.issue import Issue

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
	arg_parser.add_argument(
	    "-t",
	    "--translate",
	    action='store_true',
	    required=False,
	    help='If set, translates collected issues to english')
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
	        repository_namespace: str, translate: bool, label: str):
		self.project_origin = project_origin
		self.persistency_path = persistency_path
		self.repository_namespace = repository_namespace
		self.translate = translate
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

		new_repo = IssueRepository(self.project_origin, self.persistency_path)
		print("Parsing", end=" ")
		if self.translate:
			print("and translating", end=" ")
		print("collected issues.\nThis might take a few minutes")

		for issue in tqdm(collected_issues):
			# print(vars(issue))

			new_issue = Issue(str(issue.number),
			                  issue.title,
			                  creation_time=issue.created_at,
			                  completion_time=issue.closed_at)
			new_repo.add_issue(new_issue, warn_duplicates=True)

		return new_repo


def main():
	"""
	Main code for executing the crawler script
	"""
	args = parse_args()

	collector = GitHubCrawler(args.name, args.output, args.project,
	                          args.translate, args.label)

	resulting_issues = collector.collect_issues()

	print(len(resulting_issues.get_issues()), "issues collected")

	resulting_issues.save()


if __name__ == "__main__":
	main()
