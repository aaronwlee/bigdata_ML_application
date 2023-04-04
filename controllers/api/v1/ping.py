from flask import Blueprint, jsonify

ping = Blueprint('ping', __name__)

@ping.route('/ping')
def sayhi():
    test = {"status": 200, "data": "pong"}
    return jsonify(test)

