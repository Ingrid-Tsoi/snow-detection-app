async function upload() {
    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];
    let status = document.getElementById("status");
    let img = document.getElementById("resultImg");

    if (!file) {
        alert("Please upload a GeoTIFF file");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    try {
        status.innerText = "Uploading...";
        img.style.display = "none";

        let res = await fetch("https://snow-detection-app.onrender.com/predict", {
            method: "POST",
            body: formData
        });

        status.innerText = "Running model...";

        let blob = await res.blob();

        status.innerText = "Rendering result...";

        let url = URL.createObjectURL(blob);

        img.src = url;
        img.style.display = "block";

        status.innerText = "Done";

    } catch (error) {
        console.error(error);
        status.innerText = "Error!";
    }
}