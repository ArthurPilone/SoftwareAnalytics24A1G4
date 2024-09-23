"""
	Module responsible for containing various data parsers
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from dateutil.parser import parse

GITLAB_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
PYTHON_MS_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
GITHUB_DATE_FORMAT = "%Y-%m-%d %H:%M:%S%:z"


# def is_it_nan(val):
# 	"""
# 	Parses a given value and decides whether it is a NaN value
# 	"""

# 	try:
# 		return np.isnan(val)
# 	except TypeError:
# 		return False


def parse_datetime_str(date_str: str):
	"""
	Parses a string to a corresponding datetime object
	"""
	try:
		new_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
	except ValueError:
		try:
			new_date = datetime.strptime(date_str, GITLAB_DATE_FORMAT)
		except ValueError:
			try:
				new_date = datetime.strptime(date_str, PYTHON_MS_DATE_FORMAT)
			except ValueError:
				try:
					new_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
				except ValueError:
					new_date = parse(date_str)

	return new_date