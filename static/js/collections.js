import { upload_file } from "./modules/api_services.js"

const form = document.querySelector(".insert_by_file")
console.log(form)
form.onsubmit = async (event) => {
    event.preventDefault()
    const result = await upload_file(form)
    const upload_modal_toggle = document.querySelector("#upload_modal_toggle")
    window.location.reload()

}

const alertPlaceholder = document.getElementById('liveAlertPlaceholder')
const appendAlert = (message, type) => {
  const wrapper = document.createElement('div')
  wrapper.innerHTML = [
    `<div class="alert alert-${type} alert-dismissible" role="alert">`,
    `   <div>${message}</div>`,
    '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
    '</div>'
  ].join('')

  alertPlaceholder.append(wrapper)
}

