from flask import Blueprint, jsonify, request, current_app
from forms import *
from werkzeug.utils import secure_filename
from threading import Thread
from services.mongo_service import drop_collection

from services.spark_service import sprak_write_file_to_mongo
from utils import allowed_file, set_event, clear_event, event_is_set
import os

data_handler = Blueprint('data_handler', __name__)

@data_handler.route('/insert_by_file', methods=["POST"])
def insert_file():
    try:
        if event_is_set():
            raise Exception("A file upload operation is in progress in the background. You cannot upload files at this time.")
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file_name)
            file.save(file_path)
            # drop collection if exist
            drop_collection(file_name)
            th = Thread(target=sprak_write_file_to_mongo, args=(file_name, file_path, set_event, clear_event), daemon=True)
            th.start()

            return jsonify({"status": 200, "message": "Successfully pend a uploading in background. Please refresh the page to check."})
        else:
            return jsonify({"status": 500, "message": "The file either does not exist or is not a csv or txt file format."})
    except Exception as e:
        raise e
