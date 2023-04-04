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
    }

}

export { upload_file }