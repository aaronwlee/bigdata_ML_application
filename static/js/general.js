const alertPlaceholder = document.getElementById('liveAlertPlaceholder')
const appendAlert = (message, type) => {
  const wrapper = document.createElement('div')
  wrapper.innerHTML = [
    `<div class="alert alert-${type} fade show alert-dismissible" role="alert">`,
    `   <div>${message}</div>`,
    '   <button id="remote-close" type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
    '</div>'
  ].join('')

  alertPlaceholder.append(wrapper)
}


window.onload = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const message = urlParams.get('message');
    console.log(message)
    if (message) {
        appendAlert(message, "info")
        setTimeout(() => {
            window.history.replaceState({}, "", window.location.pathname);
            document.querySelector("#remote-close").click()
        }, 10000);
        
    }
}