#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template
import logging
from logging import Formatter, FileHandler

from dotenv import load_dotenv
import os

from database import MongoJsonEncoder

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

load_dotenv() # load .env file
app = Flask(__name__)
app.config.from_pyfile('config.py') # this config for the basic application setup

# Mongodb connection
app.config['MONGO_URI'] = os.getenv('MONGO_URI')

# For decoding the bson object
app.json_provider_class = MongoJsonEncoder

#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


#-------------------------------------------------------#
# Imports
#-------------------------------------------------------#

from controllers.view_controller import view_controller
from controllers.api.v1 import version_1_controller

#-------------------------------------------------------#
# Registers
#-------------------------------------------------------#

app.register_blueprint(view_controller)

# Version 1
app.register_blueprint(version_1_controller)

# Error handlers.
@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    print(e)
    # now you're handling non-HTTP exceptions only
    return render_template("errors/500.html", e=e), 500

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Or specify port manually:

if __name__ == '__main__':
    if app.config['PORT']:
        port = app.config['PORT']
    else:
        port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

