from flask import Blueprint, render_template, request, redirect

from services.mongo_service import get_collection_list, get_by_collection_with_page, get_collection_page_size, get_collection_expain, insert_one_to_collection
from utils import get_adjust_pagination, event_is_set

view_controller = Blueprint('view_controller', __name__, template_folder='templates')


@view_controller.route('/')
def index():
    return render_template(f'pages/home.html')
    

@view_controller.route('/collections')
def collections():
    message = request.args.get('message')
    collections = get_collection_list()
    data = { "collections": collections, "busy_collections": event_is_set(), "message": message }
    return render_template(f'pages/collections.html', data=data)

@view_controller.route('/detail/<collection>')
def detail(collection):
    page_num = request.args.get('page')
    if not page_num:
        return redirect(f"/detail/{collection}?page=1")
    page_num = int(page_num)
    data = get_by_collection_with_page(collection, page_num)
    count = get_collection_page_size(collection)
    pagination = get_adjust_pagination(count, page_num)
    fields = get_collection_expain(collection)
    result = {"title": collection, "data": data, "pagination": pagination, "current_page": page_num, "fields": fields}

    return render_template(f'pages/detail.html', result=result)

@view_controller.route('/add/<collection>', methods = ['GET', 'POST', 'DELETE'])
def add(collection):
    fields = get_collection_expain(collection)
    if request.method == 'GET':
        return render_template(f'pages/add.html', data={"fields": fields, "title": collection})
    elif request.method == 'POST':
        data_dict = {}
        for field in fields:
            if field != "_id":
                input_data = request.form.get(field)
                if input_data == '':
                    data_dict[field] = None
                else:
                    data_dict[field] = input_data
        print(data_dict)
        insert_one_to_collection(collection, data_dict)
        return redirect('/collections?message="Successfully insert a data"')
    else:
        print(request.method)
        return redirect('/collections')