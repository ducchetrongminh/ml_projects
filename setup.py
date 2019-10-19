# Standard library imports
import subprocess
import sys

# Third-party imports

# Local imports


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':
	# Install required packages when reset session
	# install('PyJWT')
    pass