// This function is now called directly by the button's onclick event
function showLoader() {
    const loaderOverlay = document.getElementById('loader-overlay');
    if (loaderOverlay) {
        loaderOverlay.style.display = 'flex';
    }
}

// All other event listeners for a better UI experience
document.addEventListener('DOMContentLoaded', function () {
    const fileUpload = document.getElementById('file-upload');
    const uploadArea = document.getElementById('upload-area');
    const filenameDisplay = document.getElementById('filename-display');

    if (uploadArea) {
        // Handle file selection via click
        fileUpload.addEventListener('change', () => {
            if (fileUpload.files.length > 0) {
                filenameDisplay.textContent = `Selected: ${fileUpload.files[0].name}`;
            } else {
                filenameDisplay.textContent = '';
            }
        });

        // Make the whole area clickable
        uploadArea.addEventListener('click', () => fileUpload.click());

        // Handle drag and drop styling
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileUpload.files = files; // Assign the dropped files to the input
                filenameDisplay.textContent = `Selected: ${files[0].name}`;
            }
        });
    }
});