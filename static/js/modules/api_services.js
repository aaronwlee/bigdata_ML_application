async function upload_file(form) {
    try {
        const response = await fetch("/api/v1/insert_by_file", {
            body: new FormData(form),
            method: "POST",
        })
        const jsonData = await response.json();
        return jsonData
    } catch (err) {
        console.error(err)
        throw err
    }

}

async function get_collections() {
    try {
        const response = await fetch("/api/v1/get_collections")
        const jsonData = await response.json();
        return jsonData
    } catch (err) {
        console.error(err)
        throw err
    }
}

async function get_thread_status() {
    try {
        const response = await fetch("/api/v1/get_thread_status")
        const jsonData = await response.json();
        return jsonData
    } catch (err) {
        console.error(err)
        throw err
    }
}

export { upload_file, get_collections, get_thread_status }