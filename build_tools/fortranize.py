#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import os
import re
import sys
import hashlib
import subprocess
import shutil
import glob

DEFAULT_ROOT = 'skutil'

# WindowsError is not defined on unix systems
try:
	WindowsError
except NameError:
	WindowsError = None


def check_and_fortranize(root_dir):
	print('Roor dir: %s' % root_dir)
	for cur_dir, dirs, files in os.walk(root_dir):

		# if any of the files in the directory end in .f compile them
		for filename in files:
			if filename.endswith('.f'):

				# extract the first portion
				canonical_name = filename.split('.')[0]
				abs_file_name  = os.path.join(cur_dir, filename)

				# construct the subprocess to call
				print('Creating %s module from %s' % (canonical_name+'.so', abs_file_name))
				subprocess.call(['f2py', '-c', '-m', canonical_name, abs_file_name])

				# get the glob
				g = glob.glob('%s*' % canonical_name)

				# move the module(s) to the cur_dir
				for module in g:
					shutil.move(module, cur_dir)

def main(root_dir=DEFAULT_ROOT):
	check_and_fortranize(root_dir)

if __name__ == '__main__':
	try:
		root_dir_arg = sys.argv[1]
	except IndexError:
		root_dir_arg = DEFAULT_ROOT
	main(root_dir_arg)