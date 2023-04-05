function create_collection_placeholders() {
    const tr = document.createElement("tr")

    const placeholders = []
    for (let i = 0; i < 3; i++) {
        const container = document.createElement("td")
        container.setAttribute("class", "placeholder-glow")
        const placeholder = document.createElement("span")
        placeholder.setAttribute("class", "placeholder col-8")
        placeholder.setAttribute("style", "height: 24px; margin: 7px 0;")
        container.appendChild(placeholder)

        placeholders.push(container)
    }
    tr.append(...placeholders, document.createElement("td"))
    return tr
}


export default function placeholder_render() {
    const placeholder_row = []
    for (let i = 0; i < 5; i++) {
      placeholder_row.push(create_collection_placeholders())
    }
    return placeholder_row
}