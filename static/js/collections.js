import alert_render from "./modules/alert_render.js"
import { upload_file, get_collections, get_thread_status, delete_collection } from "./modules/api_services.js"
import placeholder_render from "./modules/placeholder_render.js"

async function block_new_collection() {
  const upload_btn = document.querySelector("#upload_modal_toggle")

  const response = await get_thread_status()
  if (response.status == "failed") {
    // do error handling message
    return null
  }
  if (response.data == false) {
    if (upload_btn.getAttribute("disabled") != null) {
      alert_render("A file upload operation completed successfully.", "info")
    }
    upload_btn.removeAttribute("disabled")
  } else {
    alert_render("A file upload operation is in progress in the background. You cannot upload files at this time.", "warning")
    upload_btn.setAttribute("disabled", true)
  }
}

function confirm_delete(collection) {
  if (confirm("Are you sure to delete?") == true) {
    delete_collection(collection)
    reload_collections()
  }
}

async function reload_collections() {
  // disable_buttons
  await block_new_collection()

  const target = document.querySelector("#collection-target")
  // clear prev data
  target.replaceChildren(...placeholder_render())

  const response = await get_collections()
  if (response.status == "failed") {
    // do error handling message
    return null
  }
  const collections = response.data
  const new_table_nodes = collections.map((collection, i) => {
    const tr = document.createElement("tr")

    // create collection number table data
    const th = document.createElement("th")
    th.setAttribute("scope", "row")
    th.innerText = i+1

    // create collection name table data
    const collection_name = document.createElement("td")
    collection_name.setAttribute("style", "cursor: pointer;")
    collection_name.onclick = () => {
      window.location.href = `/detail/${collection}`
    }
    collection_name.innerText = collection

    // create append table data
    const appendTd = document.createElement("td")
    appendTd.setAttribute("class", "text-center")
    appendTd.innerHTML = [
      `<a role="button" class="btn btn-primary right-space" href="/add/${collection}">`,
      `<i class="bi bi-pencil"></i>`,
      `</a>`,

      `<a id="delete-collection" role="button" class="btn btn-primary right-space">`,
      `<i class="bi bi-trash3"></i>`,
      `</a>`,

      `<a id="demo" role="button" class="btn btn-primary" href="/demo/${collection}">`,
      `Demo`,
      `</a>`,
    ].join('')

    appendTd.querySelector("#delete-collection").onclick = () => confirm_delete(collection)
    

    tr.append(th, collection_name, appendTd)

    return tr
  })

  target.replaceChildren(...new_table_nodes)

}

const form = document.querySelector(".insert_by_file")
form.onsubmit = async (event) => {
    event.preventDefault()
    const result = await upload_file(form)
    const upload_modal_toggle = document.querySelector("#upload_modal_toggle")
    upload_modal_toggle.click()
    alert_render(result.data, result.status == "ok" ? "info" : "warning")
    reload_collections()
}

reload_collections()
get_thread_status()

window.reload_collections = reload_collections