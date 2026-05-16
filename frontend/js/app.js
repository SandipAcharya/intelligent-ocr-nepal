document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const extractBtn = document.getElementById('extract-btn');
    const docTypeSelect = document.getElementById('doc-type');
    
    const previewSection = document.getElementById('preview-section');
    const docPreview = document.getElementById('doc-preview');
    
    const emptyState = document.getElementById('empty-state');
    const loadingSpinner = document.getElementById('loading-spinner');
    const kycForm = document.getElementById('kyc-form');
    const rawTextContainer = document.getElementById('raw-text-container');
    const rawTextOutput = document.getElementById('raw-text-output');
    const copyTextBtn = document.getElementById('copy-text-btn');
    const submitBtn = document.getElementById('submit-btn');
    
    let selectedFile = null;

    // Handle File Selection
    fileInput.addEventListener('change', (e) => {
        if(e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if(e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        extractBtn.disabled = false;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            docPreview.src = e.target.result;
            previewSection.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Handle Extraction
    extractBtn.addEventListener('click', async () => {
        if(!selectedFile) return;

        // UI Updates
        // UI Updates
        emptyState.classList.add('hidden');
        kycForm.classList.add('hidden');
        if (rawTextContainer) rawTextContainer.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        extractBtn.disabled = true;

        const formData = new FormData();
        formData.append('document', selectedFile);
        formData.append('document_type', docTypeSelect.value);

        try {
            // Assume API runs on localhost:5000 for this prototype
            const response = await fetch('http://localhost:5000/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if(response.ok) {
                if (result.data && result.data.document_type === 'general') {
                    showRawText(result.raw_text);
                } else {
                    populateForm(result.data);
                }
            } else {
                alert('Extraction Failed: ' + result.error);
                emptyState.classList.remove('hidden');
            }
        } catch(error) {
            console.error('Error:', error);
            alert('Failed to connect to the backend server. Make sure it is running on port 5000.');
            emptyState.classList.remove('hidden');
        } finally {
            loadingSpinner.classList.add('hidden');
            extractBtn.disabled = false;
        }
    });

    function populateForm(data) {
        kycForm.classList.remove('hidden');
        
        // Fields
        document.getElementById('first_name').value = data.first_name || '';
        document.getElementById('last_name').value = data.last_name || '';
        document.getElementById('citizenship_no').value = data.citizenship_no || '';
        document.getElementById('dob').value = data.date_of_birth || '';
        document.getElementById('address').value = data.address || '';
        
        // Confidence badge
        const badge = document.getElementById('confidence-badge');
        const conf = Math.round(data.confidence_score * 100);
        badge.textContent = `Confidence: ${conf}%`;
        
        badge.className = 'badge ' + (conf > 80 ? 'high' : 'low');
        
        // Highlight low confidence fields to user (Human in the loop)
        const fields = ['citizenship_no', 'dob'];
        fields.forEach(f => {
            const el = document.getElementById(f);
            if(conf < 78 || !el.value) {
                el.classList.add('low-conf');
            } else {
                el.classList.remove('low-conf');
            }
        });

        // Face crop if available
        if (data.face_image_path) {
            document.getElementById('face-preview').src = `http://localhost:5000${data.face_image_path}`;
        }
    }

    function showRawText(rawTextArray) {
        if (rawTextContainer) {
            rawTextContainer.classList.remove('hidden');
            rawTextOutput.value = rawTextArray.join('\n');
            
            // Update confidence badge
            const badge = document.getElementById('confidence-badge');
            badge.textContent = `General Extraction`;
            badge.className = 'badge high';
        }
    }

    if (copyTextBtn) {
        copyTextBtn.addEventListener('click', () => {
            rawTextOutput.select();
            document.execCommand('copy');
            copyTextBtn.textContent = 'Copied! ✅';
            setTimeout(() => {
                copyTextBtn.textContent = 'Copy to Clipboard 📋';
            }, 2000);
        });
    }

    // Submit Action
    kycForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const payload = {
            document_type: docTypeSelect.value,
            first_name: document.getElementById('first_name').value,
            last_name: document.getElementById('last_name').value,
            citizenship_no: document.getElementById('citizenship_no').value,
            date_of_birth: document.getElementById('dob').value,
            address: document.getElementById('address').value
        };

        submitBtn.disabled = true;
        submitBtn.textContent = 'Submitting...';

        try {
            const response = await fetch('http://localhost:5000/api/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if(response.ok) {
                alert('Record verified and submitted successfully!');
                // Reset UI
                kycForm.classList.add('hidden');
                emptyState.classList.remove('hidden');
                previewSection.classList.add('hidden');
                fileInput.value = '';
                selectedFile = null;
            } else {
                alert('Submission Failed.');
            }
        } catch(error) {
            console.error(error);
            alert('Failed to connect to the backend server.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Verify & Submit ✅';
        }
    });

    // Reset Form
    document.getElementById('reset-btn').addEventListener('click', () => {
        document.getElementById('first_name').value = '';
        document.getElementById('last_name').value = '';
        document.getElementById('citizenship_no').value = '';
        document.getElementById('dob').value = '';
        document.getElementById('address').value = '';
        document.querySelectorAll('.low-conf').forEach(el => el.classList.remove('low-conf'));
    });
});
