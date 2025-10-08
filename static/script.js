document.addEventListener('DOMContentLoaded', () => {
    // --- Get all the HTML elements we need ---
    const dropZone = document.getElementById('drop-zone');
    const photoInput = document.getElementById('photo-input');
    const uploadButton = document.getElementById('upload-button');
    const statusText = document.getElementById('status-text');
    const progressBar = document.getElementById('progress-bar');
    const previewArea = document.getElementById('preview-area');
    const resultsArea = document.getElementById('results-area');

    let selectedFiles = []; // This will store the files to be uploaded

    // --- Drag & Drop and File Selection Logic ---
    dropZone.addEventListener('click', () => photoInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    photoInput.addEventListener('change', () => handleFiles(photoInput.files));

    function handleFiles(files) {
        selectedFiles = [...files];
        previewArea.innerHTML = '';
        resultsArea.innerHTML = '<p>Your sorted photos will appear here.</p>'; // Reset results on new selection

        if (selectedFiles.length === 0) {
            uploadButton.disabled = true;
            statusText.innerText = '';
            return;
        }

        uploadButton.disabled = false;
        statusText.innerText = `${selectedFiles.length} file(s) selected. Ready to process.`;

        // Generate image previews
        for (const file of selectedFiles) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.title = file.name;
                previewArea.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    }

    // --- Upload Logic ---
    uploadButton.addEventListener('click', async () => {
        if (selectedFiles.length === 0) {
            statusText.innerText = 'Please select some photos first.';
            return;
        }

        statusText.innerText = 'Uploading photos...';
        uploadButton.disabled = true;

        const formData = new FormData();
        for (const file of selectedFiles) {
            formData.append('photos', file);
        }

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if (response.ok) {
                pollJobStatus(data.job_id); // Start checking the job status
            } else {
                statusText.innerText = `Error: ${data.error}`;
                uploadButton.disabled = false;
            }
        } catch (error) {
            statusText.innerText = `An error occurred: ${error}`;
            uploadButton.disabled = false;
        }
    });

    // --- Polling and Display Logic ---
    const pollJobStatus = (jobId) => {
        progressBar.style.display = 'block';
        statusText.innerText = 'Processing... 0%';

        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/api/status/${jobId}`);
                const data = await response.json();

                if (data.progress !== undefined) {
                    progressBar.value = data.progress;
                    statusText.innerText = `Processing... ${data.progress}%`;
                }

                if (data.status === 'complete') {
                    clearInterval(interval);
                    progressBar.value = 100;
                    statusText.innerText = 'Processing complete! Displaying results.';
                    displayResults(data.result);
                    uploadButton.disabled = false;
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    statusText.innerText = `Job failed: ${data.error}`;
                    uploadButton.disabled = false;
                }
            } catch (error) {
                clearInterval(interval);
                statusText.innerText = 'Error checking status.';
                uploadButton.disabled = false;
            }
        }, 2000); // Check every 2 seconds
    };

    const displayResults = (results) => {
        resultsArea.innerHTML = '';
        const clusters = {};
        if (!results || typeof results === 'string') {
            resultsArea.innerText = results || 'No results to display.';
            return;
        }
        for (const face of results) {
            if (!clusters[face.cluster_id]) {
                clusters[face.cluster_id] = [];
            }
            if (!clusters[face.cluster_id].includes(face.image_url)) {
                 clusters[face.cluster_id].push(face.image_url);
            }
        }
        for (const clusterId in clusters) {
            const clusterDiv = document.createElement('div');
            const title = document.createElement('h4');
            title.innerText = clusterId === '-1' ? 'Unknowns' : `Person ${parseInt(clusterId) + 1}`;
            clusterDiv.appendChild(title);
            const imageContainer = document.createElement('div');
            clusters[clusterId].forEach(imageUrl => {
                const img = document.createElement('img');
                img.src = imageUrl;
                img.style.width = '150px';
                img.style.margin = '5px';
                imageContainer.appendChild(img);
            });
            clusterDiv.appendChild(imageContainer);
            resultsArea.appendChild(clusterDiv);
        }
    };
});