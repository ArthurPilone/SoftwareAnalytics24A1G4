"""
	Module responsible for housing the Issue class definition.
"""

import sys
import os
import ast
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(os.path.abspath('')).absolute()))

from parsers import parse_datetime_str, is_it_nan


class Issue:
	"""
		Represents a single issue extracted from an issue tracker
		tool.
	"""

	# Meta data may include:
	## url
	## Collection date
	## author_association

	def __init__(  #pylint: disable=too-many-arguments, dangerous-default-value
	        self,
	        identifier: str,
			assignee: str,
	        summary: str,
	        body: str = "",
	        creation_time: None | datetime = None,
	        completion_time: None | datetime = None,
	        extra_data: dict = {}):
		self.identifier = identifier

		self.assignee = assignee

		self.summary = summary
		self.body = body

		self.creation_time = creation_time
		self.completion_time = completion_time

		self.extra_data = extra_data

	@staticmethod
	def from_dict(dict_obj: dict):  #pylint: disable=too-many-branches
		"""
			Creates a new Issue object from a python dictionary
		"""

		new_id, new_summary, assignee = "", "", None
		creation_time, completion_time = None, None
		body = ""
		remaining_data = {}

		for key in dict_obj.keys():  #pylint: disable=duplicate-code
			val = dict_obj[key]

			if is_it_nan(val):
				continue

			if key in ("id", "identifier", "id\r"):
				new_id = str(val)
			elif key in ("summary", "title"):
				new_summary = val
			elif key in ("body", "description"):
				body = val  #pylint: disable=duplicate-code
			elif key in ("assignee", "author"):
				assignee = val
			elif key in (
			    "creation_time",
			    "time",
			    "created_at",
			):
				creation_time = parse_datetime_str(val)
			elif key in ("remaining_data", "extra_data"):
				value_loaded = ast.literal_eval(val)
				remaining_data.update(value_loaded)
			elif key in ("completion_time", "closed_at"):
				if str(val) not in ("NaN", "nan"):
					completion_time = parse_datetime_str(val)
			else:
				remaining_data[key] = val

		if new_id == "":
			print(dict_obj)
			raise ValueError("Tried to create issue from dict with no 'id' key")
		if new_summary == "":
			raise ValueError(
			    "Tried to create issue from dict with no 'title'/'summary' key")

		new_issue = Issue(new_id,
						  assignee,
		                  new_summary,
		                  body=body,
		                  creation_time=creation_time,
		                  completion_time=completion_time,
		                  extra_data=remaining_data,
		                  )

		return new_issue

	def to_dict(self) -> dict:
		"""
		Returns a dictionary containing this issue's properties.
		"""
		return vars(self)

	def get_id(self) -> str:
		"""
			Returns the issue's identifier
		"""
		return self.identifier

	def get_summary(self) -> str:
		"""
			Returns the issue's summary (or title)
		"""
		return self.summary
