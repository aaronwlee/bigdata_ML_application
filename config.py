import os

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

# Enable debug mode.
DEBUG = True
UPLOAD_FOLDER = './temp'
PORT = 80

SECRET_KEY = 'non-secret-key'
