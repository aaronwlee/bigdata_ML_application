import { upload_file } from "./modules/api_services.js"

function prepare_for_file_upload() {
    form = document.querySelector(".insert_by_file")
    if (form) {
        console.log(form)
    }
}


// Register into window object
window.prepare_for_file_upload = prepare_for_file_upload