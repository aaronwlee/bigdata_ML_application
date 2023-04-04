from flask import Blueprint, jsonify
from forms import *

ping = Blueprint('ping', __name__)

@ping.route('/ping')
def sayhi():
    test = {"status": 200, "data": "pong"}
    return jsonify(test)

