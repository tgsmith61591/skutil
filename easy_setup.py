import subprocess, sys, os

def easy_setup():
	cwd = os.path.abspath(os.path.dirname(__file__))
	subprocess.call([sys.executable, 'setup.py', 'install']) # do installation

if __name__ == '__main__':
	easy_setup()
