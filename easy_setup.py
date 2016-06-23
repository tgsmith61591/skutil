import subprocess, sys, os

def easy_setup():
	cwd = os.path.abspath(os.path.dirname(__file__))
	# cleans in this step:
	subprocess.call([sys.executable, 'setup.py', 'build_ext', '--inplace'])
	subprocess.call([sys.executable, 'setup.py', 'install']) # do installation

if __name__ == '__main__':
	easy_setup()
