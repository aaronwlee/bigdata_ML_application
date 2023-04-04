import { upload_file } from "./modules/api_services.js"

const form = document.querySelector(".insert_by_file")
console.log(form)
form.onsubmit = async (event) => {
    event.preventDefault()
    const result = await upload_file(form)
    const upload_modal_toggle = document.querySelector("#upload_modal_toggle")
    window.location.href = "/collections?message=Successfully uploaded"
}

