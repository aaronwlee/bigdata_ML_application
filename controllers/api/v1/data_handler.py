from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from threading import Thread
from services.mongo_service import drop_collection, get_collection_list

from services.spark_service import sprak_write_file_to_mongo
from utils import allowed_file, set_event, clear_event, event_is_set
import os

data_handler = Blueprint('data_handler', __name__)

@data_handler.route('/insert_by_file', methods=["POST"])
def insert_by_file():
    try:
        if event_is_set():
            raise Exception("A file upload operation is in progress in the background. You cannot upload files at this time.")
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            try:
                os.mkdir(current_app.config['UPLOAD_FOLDER'])
            except Exception:
                print("Folder existed, good to go")
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file_name)
            file.save(file_path)
            th = Thread(target=sprak_write_file_to_mongo, args=(file_name, file_path, set_event, clear_event), daemon=True)
            th.start()

            return jsonify({"status": "ok", "data": "Successfully pend a uploading in background. Please refresh the page to check."})
        else:
            return jsonify({"status": "failed", "data": "The file either does not exist or is not a csv or txt file format."})
    except Exception as e:
        raise e
    
@data_handler.route('/get_collections', methods=["GET"])
def get_collections():
    try:
        collections = get_collection_list()
        result = {
            "status":"ok",
            "data": collections
        }
        return jsonify(result)
    except Exception as e:
        raise e

@data_handler.route('/get_thread_status', methods=["GET"])
def get_thread_status():
    result = {
        "status": "ok",
        "data": event_is_set()
    }
    return jsonify(result)

@data_handler.route('/delete_collection', methods=["DELETE"])
def delete_collection():
    body = request.json
    print(body)
    if not body.get("collection"):
        raise Exception("Unable to find collection in request")
    
    drop_collection(body.get("collection"))
    result = {
        "status": "ok",
    }
    return jsonify(result)
