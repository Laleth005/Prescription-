// Common UI Enhancements for Prescription Analyzer

document.addEventListener('DOMContentLoaded', function() {
    // Enable file drag and drop behavior
    setupFileDragAndDrop();
    
    // Add responsive navigation for mobile
    setupMobileNavigation();
});

function setupFileDragAndDrop() {
    const fileUpload = document.querySelector('.file-upload');
    const fileInput = document.querySelector('input[type="file"]');
    const placeholder = document.querySelector('.file-upload-placeholder');
    
    if (!fileUpload || !fileInput || !placeholder) return;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileUpload.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        fileUpload.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        fileUpload.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        placeholder.classList.add('highlight');
    }
    
    function unhighlight() {
        placeholder.classList.remove('highlight');
    }
    
    fileUpload.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            fileInput.files = files;
            updatePreview(files[0]);
        }
    }
    
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            updatePreview(this.files[0]);
        }
    });
    
    function updatePreview(file) {
        const preview = document.getElementById('file-preview');
        if (!preview) return;
        
        preview.innerHTML = '';
        
        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.classList.add('file-preview-image');
            img.file = file;
            preview.appendChild(img);
            
            const reader = new FileReader();
            reader.onload = (function(aImg) { 
                return function(e) { 
                    aImg.src = e.target.result; 
                }; 
            })(img);
            reader.readAsDataURL(file);
        } else {
            const fileIcon = document.createElement('i');
            fileIcon.className = 'fas fa-file-pdf';
            preview.appendChild(fileIcon);
            
            const fileName = document.createElement('p');
            fileName.textContent = file.name;
            preview.appendChild(fileName);
        }
    }
}

function setupMobileNavigation() {
    const header = document.querySelector('header');
    
    if (!header) return;
    
    // Check if navigation toggle already exists
    if (document.querySelector('.nav-toggle')) return;
    
    const nav = document.querySelector('nav');
    
    if (!nav) return;
    
    // Create mobile navigation toggle
    const navToggle = document.createElement('button');
    navToggle.className = 'nav-toggle';
    navToggle.innerHTML = '<i class="fas fa-bars"></i>';
    navToggle.setAttribute('aria-label', 'Toggle navigation');
    
    // Add toggle to header
    header.insertBefore(navToggle, nav);
    
    // Toggle navigation on click
    navToggle.addEventListener('click', function() {
        nav.classList.toggle('nav-open');
        
        if (nav.classList.contains('nav-open')) {
            navToggle.innerHTML = '<i class="fas fa-times"></i>';
        } else {
            navToggle.innerHTML = '<i class="fas fa-bars"></i>';
        }
    });
    
    // Add mobile navigation styles
    const styleEl = document.createElement('style');
    styleEl.textContent = `
        @media (max-width: 768px) {
            header {
                position: relative;
            }
            
            .nav-toggle {
                display: block;
                background: none;
                border: none;
                font-size: 1.5rem;
                color: var(--primary-color);
                cursor: pointer;
            }
            
            nav {
                display: none;
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background-color: white;
                box-shadow: var(--shadow);
                flex-direction: column;
                padding: 1rem;
                z-index: 100;
            }
            
            nav.nav-open {
                display: flex;
            }
        }
        
        @media (min-width: 769px) {
            .nav-toggle {
                display: none;
            }
        }
    `;
    document.head.appendChild(styleEl);
}
