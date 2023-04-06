function _appendBuffer(buffer1, buffer2) {
    var tmp = new Uint8Array(buffer1.byteLength + buffer2.byteLength);
    tmp.set(new Uint8Array(buffer1), 0);
    tmp.set(new Uint8Array(buffer2), buffer1.byteLength);
    return tmp.buffer;
}

function scrollToBottom() {
    setTimeout(() => {
        window.scrollTo(0, document.body.scrollHeight);
    }, 100)
}

function predict() {
    const socket = io();
    const target = document.querySelector("#ml-progress")

    let image_chunk = new Uint8Array()


    socket.on("connect", () => {
        console.log(socket.id); // x8WIv7-mJelg7on_ALbx

        const collection_name = window.location.pathname.split("/")
        socket.emit("start", collection_name.at(-1))
    });
    
    socket.on("message", (arg) => {
        const box = document.createElement("div")
        box.setAttribute("class", "fancy-box")
        box.innerHTML = [
            `<div class="card border-primary">`,
            `<h5 class="card-header">Pyspark response</h5>`,
            `<div class="card-body">`,
            `</div>`,
            `</div>`,
        ].join('')

        const paragraph = document.createElement("p")
        paragraph.innerText = arg
        box.querySelector(".card-body").append(paragraph)

        target.append(box)
        scrollToBottom()
    });
    
    socket.on("disconnect", () => {
        console.log(socket.id); // undefined
    });

    socket.on("html", (msg) => {
        const box = document.createElement("div")
        box.setAttribute("class", "fancy-box")
        box.innerHTML = [
            `<div class="card border-primary">`,
            `<h5 class="card-header">Pyspark response</h5>`,
            `<div class="card-body">`,
            `</div>`,
            `</div>`,
        ].join('')

        const container = document.createElement("div")
        container.setAttribute("class", "table-responsive")
        container.innerHTML = msg
        const table = container.querySelector("table")
        table.removeAttribute("border")
        table.setAttribute("class", "table")
        
        box.querySelector(".card-body").append(container)
        target.append(box)
        scrollToBottom()
    });

    socket.on('img-stream', function ({name, status, data}) {

        if (status == "ongoing") {
            image_chunk = _appendBuffer(image_chunk, data)
        } else {
            const box = document.createElement("div")
            box.setAttribute("class", "fancy-box")
            box.innerHTML = [
                `<div class="card border-primary">`,
                `<h5 class="card-header">Pyspark response</h5>`,
                `<div class="card-body flex-center">`,
                `</div>`,
                `</div>`,
            ].join('')

            const img = document.createElement('img')
            var blob = new Blob([image_chunk], { type: "image/png" } );
    
            var img_url = URL.createObjectURL(blob);
            img.setAttribute('src', img_url);
            img.setAttribute("class", "predict-result")
    
            box.querySelector(".card-body").append(img)
            target.append(box)
            image_chunk = new Uint8Array()
            scrollToBottom()
        }
    });

    window.sio = socket
}

predict()




