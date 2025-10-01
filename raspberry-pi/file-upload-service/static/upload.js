// File upload handling
const uploadArea = document.querySelector('.upload-area');
const fileInput = document.getElementById('fileInput');
const statusArea = document.getElementById('statusArea');
const progressBar = document.getElementById('progressBar');
const progressFill = document.getElementById('progressFill');

// Drag and drop events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    handleFiles(files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    for (let file of files) {
        uploadFile(file);
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showStatus('Uploading ' + file.name + '...', 'loading');
    showProgress(0);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideProgress();
        
        if (data.success) {
            showStatus('✅ Successfully uploaded: ' + file.name, 'success');
        } else {
            showStatus('❌ Upload failed: ' + data.error, 'error');
        }
    })
    .catch(error => {
        hideProgress();
        showStatus('❌ Network error: ' + error.message, 'error');
    });
}

function testConnection() {
    showStatus('Testing connection...', 'loading');
    
    fetch('/test-connection')
    .then(response => response.json())
    .then(data => {
        const statusEl = document.getElementById('connectionStatus');
        
        if (data.success) {
            showStatus('✅ Connection test successful!', 'success');
            statusEl.textContent = 'Connected';
            statusEl.className = 'connection-status status-connected';
        } else {
            showStatus('❌ Connection failed: ' + data.error, 'error');
            statusEl.textContent = 'Disconnected';
            statusEl.className = 'connection-status status-disconnected';
        }
    })
    .catch(error => {
        showStatus('❌ Connection test error: ' + error.message, 'error');
        document.getElementById('connectionStatus').textContent = 'Error';
    });
}

function showStatus() {
    fetch('/status')
    .then(response => response.json())
    .then(data => {
        const info = `
            📊 Service Statistics:
            • Uptime: ${data.uptime_formatted}
            • Uploads attempted: ${data.stats.uploads_attempted}
            • Successful: ${data.stats.uploads_successful}
            • Failed: ${data.stats.uploads_failed}
            • Data transferred: ${(data.stats.total_bytes_transferred / 1024).toFixed(1)} KB
        `;
        showStatusMessage(info, 'success');
    })
    .catch(error => {
        showStatusMessage('❌ Failed to get status: ' + error.message, 'error');
    });
}

function showStatusMessage(message, type) {
    statusArea.innerHTML = message.replace(/\n/g, '<br>');
    statusArea.className = 'status-area ' + type;
    statusArea.style.display = 'block';
}

// Alias for backwards compatibility
function showStatus(message, type) {
    if (arguments.length === 0) {
        // Call the status fetching function
        fetch('/status')
        .then(response => response.json())
        .then(data => {
            const info = `
                📊 Service Statistics:
                • Uptime: ${data.uptime_formatted}
                • Uploads attempted: ${data.stats.uploads_attempted}
                • Successful: ${data.stats.uploads_successful}
                • Failed: ${data.stats.uploads_failed}
                • Data transferred: ${(data.stats.total_bytes_transferred / 1024).toFixed(1)} KB
            `;
            showStatusMessage(info, 'success');
        })
        .catch(error => {
            showStatusMessage('❌ Failed to get status: ' + error.message, 'error');
        });
    } else {
        // Show the message
        showStatusMessage(message, type);
    }
}

function showProgress(percent) {
    progressBar.style.display = 'block';
    progressFill.style.width = percent + '%';
}

function hideProgress() {
    progressBar.style.display = 'none';
    progressFill.style.width = '0%';
}

// Auto-test connection on page load
document.addEventListener('DOMContentLoaded', testConnection);
