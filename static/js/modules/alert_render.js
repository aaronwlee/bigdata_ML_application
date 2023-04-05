
export default function alert_render(message, type) {
    const alertPlaceholder = document.getElementById('liveAlertPlaceholder')

    if (alertPlaceholder.childElementCount > 2) {
        alertPlaceholder.removeChild(alertPlaceholder.firstChild)
    }

    const wrapper = document.createElement('div')
    wrapper.setAttribute("class", `alert alert-${type} fade show alert-dismissible`)
    wrapper.setAttribute("role", "alert")
    wrapper.innerHTML = [
      `   <div>${message}</div>`,
      '   <button id="remote-close" type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
    ].join('')
  
    alertPlaceholder.append(wrapper)
}
