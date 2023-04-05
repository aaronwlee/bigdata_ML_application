import alert_render from "./modules/alert_render.js"


window.onload = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const message = urlParams.get('message');
    console.log(message)
    if (message) {
        alert_render(message, "info")
        window.history.replaceState({}, "", window.location.pathname);
    }
}