import threading
from flask import g


ALLOWED_EXTENSIONS = {'txt', 'csv'}

def allowed_file(filename):
    """
    Only allow the txt and csv file
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_adjust_pagination(max, page_num):
    if max > 8:
        pagination = [1]
        if page_num < 5:
            for i in range(1, 7):
                pagination.append(i+1)
            pagination.append("...")
        elif page_num > max-5:
            pagination.append("...")
            for i in range(max-7, max-1):
                pagination.append(i+1)
        else:
            pagination.append("...")
            for i in range(page_num-3, page_num+2):
                pagination.append(i+1)
            pagination.append("...")
        pagination.append(max)
    else:
        pagination = []
        for i in range(max):
            pagination.append(i)

    return pagination

event = threading.Event()

def set_event():
    event.set()

def clear_event():
    event.clear()

def event_is_set():
    return event.is_set()