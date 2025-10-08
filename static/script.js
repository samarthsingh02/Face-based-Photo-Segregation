document.addEventListener('DOMContentLoaded', () => {
    const photoInput = document.getElementById('photo-input');
    const uploadButton = document.getElementById('upload-button');
    const statusArea = document.getElementById('status-area');
    const resultsArea = document.getElementById('results-area');

    uploadButton.addEventListener('click', async () => {
        const files = photoInput.files;
        if (files.length === 0) {
            statusArea.innerText = 'Please select some photos first.';
            return;
        }

        statusArea.innerText = 'Uploading photos...';
        const formData = new FormData();
        for (const file of files) {
            formData.append('photos', file);
        }

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if (response.ok) {
                statusArea.innerText = `Processing started! Checking status...`;
                // Start polling for the job status
                pollJobStatus(data.job_id);
            } else {
                statusArea.innerText = `Error: ${data.error}`;
            }
        } catch (error) {
            statusArea.innerText = `An error occurred: ${error}`;
        }
    });

    const pollJobStatus = (jobId) => {
        const interval = setInterval(async () => {
            const response = await fetch(`/api/status/${jobId}`);
            const data = await response.json();

            if (data.status === 'complete') {
                clearInterval(interval);
                statusArea.innerText = 'Processing complete! Displaying results.';
                displayResults(data.result);
            } else if (data.status === 'failed') {
                clearInterval(interval);
                statusArea.innerText = `Job failed: ${data.error}`;
            } else {
                statusArea.innerText = 'Still processing... Please wait.';
            }
        }, 3000); // Check every 3 seconds
    };

    const displayResults = (results) => {
        resultsArea.innerHTML = ''; // Clear the results area

        // Group images by cluster ID
        const clusters = {};
        for (const face of results) {
            if (!clusters[face.cluster_id]) {
                clusters[face.cluster_id] = [];
            }
            // Avoid adding duplicate images to a cluster
            if (!clusters[face.cluster_id].includes(face.image_url)) {
                 clusters[face.cluster_id].push(face.image_url);
            }
        }

        // Create HTML for each cluster
        for (const clusterId in clusters) {
            const clusterDiv = document.createElement('div');
            clusterDiv.className = 'cluster';

            const title = document.createElement('h4');
            title.innerText = clusterId === '-1' ? 'Unknowns' : `Person ${parseInt(clusterId) + 1}`;
            clusterDiv.appendChild(title);

            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-container';
            clusters[clusterId].forEach(imageUrl => {
                const img = document.createElement('img');
                img.src = imageUrl;
                img.width = 150; // Simple styling
                imageContainer.appendChild(img);
            });

            clusterDiv.appendChild(imageContainer);
            resultsArea.appendChild(clusterDiv);
        }
    };
});