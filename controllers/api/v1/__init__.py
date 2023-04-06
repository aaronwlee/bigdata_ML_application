from flask import Blueprint
from controllers.api.v1.ping import ping
from controllers.api.v1.data_handler import data_handler
from controllers.api.v1.demo_handler import *

version_1_controller = Blueprint('version_1_controller', __name__, url_prefix="/api/v1")
version_1_controller.register_blueprint(ping)
version_1_controller.register_blueprint(data_handler)