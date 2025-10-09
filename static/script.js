document.addEventListener('DOMContentLoaded', () => {
    // Get all the HTML elements we need
    const dropZone = document.getElementById('drop-zone');
    const photoInput = document.getElementById('photo-input');
    const uploadButton = document.getElementById('upload-button');
    const statusText = document.getElementById('status-text');
    const progressBar = document.getElementById('progress-bar');
    const previewArea = document.getElementById('preview-area');
    const resultsArea = document.getElementById('results-area');
    const presetSelector = document.getElementById('preset-selector');
    const togglePreviewsButton = document.getElementById('toggle-previews-button');

    let selectedFiles = [];

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
        resultsArea.innerHTML = '<p>Your sorted photos will appear here.</p>';

        if (selectedFiles.length === 0) {
            uploadButton.disabled = true;
            statusText.innerText = '';
            togglePreviewsButton.style.display = 'none';
            return;
        }

        uploadButton.disabled = false;
        statusText.innerText = `${selectedFiles.length} file(s) selected. Ready to process.`;

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

        if (selectedFiles.length > 20) { // Number of images before "Show All" appears
            togglePreviewsButton.style.display = 'block';
            togglePreviewsButton.innerText = `Show All (${selectedFiles.length} images)`;
            previewArea.classList.remove('show-all');
        } else {
            togglePreviewsButton.style.display = 'none';
        }
    }

    togglePreviewsButton.addEventListener('click', () => {
        previewArea.classList.toggle('show-all');
        togglePreviewsButton.innerText = previewArea.classList.contains('show-all')
            ? 'Show Less'
            : `Show All (${selectedFiles.length} images)`;
    });

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
        formData.append('preset', presetSelector.value);

        try {
            const response = await fetch('/api/process', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok) {
                pollJobStatus(data.job_id);
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
                    // --- CHANGE: Pass the jobId to the displayResults function ---
                    displayResults(data.result, jobId);
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
        }, 2000);
    };

    const displayResults = (results, jobId) => {
        resultsArea.innerHTML = '';
        if (!results || typeof results === 'string' || results.length === 0) {
            resultsArea.innerText = typeof results === 'string' ? results : 'No results to display.';
            return;
        }

        const clusters = {};
        // 1. Group all faces by their cluster ID (which can be a name or a number)
        for (const face of results) {
            if (!clusters[face.cluster_id]) {
                clusters[face.cluster_id] = {
                    images: new Set(),
                    face_ids: [],
                    is_named: face.is_named
                };
            }
            if (face.face_id) {
                clusters[face.cluster_id].face_ids.push(face.face_id);
            }
            clusters[face.cluster_id].images.add(face.image_url);
        }

        // 2. Get the cluster keys and sort them so named people appear first
        const sortedClusterIds = Object.keys(clusters).sort((a, b) => {
            const clusterA = clusters[a];
            const clusterB = clusters[b];

            if (clusterA.is_named && !clusterB.is_named) return -1; // A comes first
            if (!clusterA.is_named && clusterB.is_named) return 1;  // B comes first

            if (clusterA.is_named && clusterB.is_named) return a.localeCompare(b); // Sort alphabetically

            if (parseInt(a) === -1) return 1; // Put "Unknowns" at the very end
            if (parseInt(b) === -1) return -1;
            return parseInt(a) - parseInt(b); // Sort numerically
        });

        // 3. Loop through the SORTED keys to create the display
        for (const clusterId of sortedClusterIds) {
            const clusterData = clusters[clusterId];
            const clusterDiv = document.createElement('div');
            clusterDiv.className = 'cluster';

            // --- NEW: Title container for inline editing ---
            const titleContainer = document.createElement('div');
            titleContainer.style.display = 'flex';
            titleContainer.style.alignItems = 'center';
            titleContainer.style.gap = '10px';
            titleContainer.style.marginBottom = '10px';

            const title = document.createElement('h4');
            title.style.margin = '0';
            if (clusterData.is_named) {
                title.innerText = clusterId;
            } else {
                // FIX 1: Changed "New Person" to just "Person"
                title.innerText = clusterId === -1 || clusterId === '-1' ? 'Unknowns' : `Person ${parseInt(clusterId) + 1}`;
            }
            titleContainer.appendChild(title);

            // --- NEW: Add an editable icon for all groups except "Unknowns" ---
            if (clusterId !== -1 && clusterId !== '-1') {
                const editIcon = document.createElement('span');
                editIcon.style.cursor = 'pointer';
                editIcon.title = 'Edit Name';
                editIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M12.146.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1 0 .708l-10 10a.5.5 0 0 1-.168.11l-5 2a.5.5 0 0 1-.65-.65l2-5a.5.5 0 0 1 .11-.168l10-10zM11.207 2.5 13.5 4.793 14.793 3.5 12.5 1.207 11.207 2.5zm1.586 3L10.5 3.207 4 9.707V10h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.293l6.5-6.5zm-9.761 5.175-.106.106-1.528 3.821 3.821-1.528.106-.106A.5.5 0 0 1 5 12.5V12h-.5a.5.5 0 0 1-.5-.5V11h-.5a.5.5 0 0 1-.5-.5V10h-.5a.5.5 0 0 1-.5-.5V9.5a.5.5 0 0 1-.28-.471z"/></svg>`;                titleContainer.appendChild(editIcon);

                // --- Add these lines to style the icon ---
                editIcon.style.width = '15px';
                editIcon.style.height = '15px';
                editIcon.style.opacity = '0.6';
                editIcon.style.verticalAlign = 'middle';

                // Optional: Make the icon brighter on hover
                editIcon.addEventListener('mouseover', () => { editIcon.style.opacity = '1'; });
                editIcon.addEventListener('mouseout', () => { editIcon.style.opacity = '0.6'; });


                editIcon.addEventListener('click', () => {
                    const currentName = title.innerText;
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.value = clusterData.is_named ? currentName : '';
                    input.placeholder = currentName;

                    title.style.display = 'none';
                    editIcon.style.display = 'none';
                    titleContainer.insertBefore(input, title);
                    input.focus();

                    const saveChanges = async () => {
                        const newName = input.value.trim();
                        if (!newName || newName === currentName) {
                            titleContainer.removeChild(input);
                            title.style.display = 'block';
                            editIcon.style.display = 'block';
                            return;
                        }

                        try {
                            const response = await fetch('/api/name_cluster', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ face_ids: clusterData.face_ids, name: newName })
                            });
                            const data = await response.json();
                            if (data.success) {
                                title.innerText = newName;
                                clusterData.is_named = true;
                            } else {
                                alert(`Error: ${data.error || 'Unknown error'}`);
                            }
                        } catch (err) {
                            alert(`An error occurred: ${err}`);
                        } finally {
                            if (titleContainer.contains(input)) {
                                titleContainer.removeChild(input);
                            }
                            title.style.display = 'block';
                            editIcon.style.display = 'block';
                        }
                    };
                    input.addEventListener('blur', saveChanges);
                    input.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter') e.preventDefault(), saveChanges();
                        if (e.key === 'Escape') input.value = currentName, saveChanges();
                    });
                });
            }
            clusterDiv.appendChild(titleContainer);

            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-container';
            clusterData.images.forEach(imageUrl => {
                const img = document.createElement('img');
                img.src = imageUrl;
                imageContainer.appendChild(img);
            });
            clusterDiv.appendChild(imageContainer);
            resultsArea.appendChild(clusterDiv);
        }
    };
});