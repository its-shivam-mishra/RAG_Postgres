document.addEventListener('DOMContentLoaded', async () => {
    // Check Authentication First
    try {
        const authResponse = await fetch('/api/auth/me', {
            cache: 'no-store',
            headers: { 'Cache-Control': 'no-cache' }
        });
        if (authResponse.ok) {
            const data = await authResponse.json();
            document.getElementById('login-overlay').style.display = 'none';
            if (document.getElementById('user-info')) {
                document.getElementById('user-info').textContent = data.user.name || data.user.email || 'Authed User';
            }
            fetchDocuments();
        } else {
            // Leave the login overlay visible
        }
    } catch (error) {
        console.error("Auth check failed", error);
    }
    
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const fileList = document.getElementById('file-list');
    const chatForm = document.getElementById('chat-form');
    const queryInput = document.getElementById('query-input');
    const chatHistory = document.getElementById('chat-history');
    
    let activeDocument = null;

    async function fetchDocuments() {
        try {
            const resp = await fetch('/api/documents');
            if (resp.ok) {
                const data = await resp.json();
                fileList.innerHTML = '';
                data.documents.forEach(doc => addToFileList(doc));
            }
        } catch (e) {
            console.error("Failed to fetch documents", e);
        }
    }

    // Handle Drag & Drop
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary)';
        uploadArea.style.backgroundColor = 'rgba(99, 102, 241, 0.05)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border-color)';
        uploadArea.style.backgroundColor = 'rgba(255,255,255,0.02)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        uploadArea.style.backgroundColor = 'rgba(255,255,255,0.02)';
        
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.endsWith('.pdf') && !file.name.endsWith('.txt')) {
            showStatus('Only PDF and TXT files are allowed.', 'error');
            return;
        }

        showStatus(`Uploading ${file.name}...`, 'loading');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                showStatus(`Indexed ${data.chunks} chunks from ${file.name}`, 'success');
                addToFileList(file.name);
            } else {
                showStatus(data.detail || 'Upload failed', 'error');
            }
        } catch (error) {
            showStatus(`Error: ${error.message}`, 'error');
        }
    }

    function showStatus(message, type) {
        uploadStatus.textContent = message;
        uploadStatus.className = `upload-status status-${type}`;
    }

    function addToFileList(filename) {
        const existing = Array.from(fileList.children).find(li => li.dataset.filename === filename);
        if (existing) return;

        const li = document.createElement('li');
        li.textContent = filename;
        li.dataset.filename = filename;
        
        li.addEventListener('click', () => {
            Array.from(fileList.children).forEach(child => child.classList.remove('active'));
            li.classList.add('active');
            activeDocument = filename;
        });

        fileList.appendChild(li);
        
        // Auto-select if first document
        if (!activeDocument) {
            li.click();
        }
    }

    // Chat functionality
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const question = queryInput.value.trim();
        if (!question) return;

        // Add user message
        appendMessage('user', question);
        queryInput.value = '';

        // Add typing indicator
        const typingId = appendTypingIndicator();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, filename: activeDocument })
            });

            const data = await response.json();
            removeMessage(typingId);

            if (response.ok) {
                appendMessage('ai', data.answer, data.sources);
            } else {
                appendMessage('ai', `Error: ${data.detail || 'Failed to get answer'}`);
            }
        } catch (error) {
            removeMessage(typingId);
            appendMessage('ai', `Connection error: ${error.message}`);
        }
    });

    function appendMessage(role, content, sources = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = role === 'ai' ? '<i data-feather="cpu"></i>' : '<i data-feather="user"></i>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources-container';
            sourcesDiv.textContent = 'Sources: ';
            
            sources.forEach(src => {
                const badge = document.createElement('span');
                badge.className = 'sources-badge';
                badge.textContent = src;
                sourcesDiv.appendChild(badge);
            });
            
            contentDiv.appendChild(sourcesDiv);
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        
        feather.replace();
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function appendTypingIndicator() {
        const id = 'typing-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message';
        msgDiv.id = id;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = '<i data-feather="cpu"></i>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        
        feather.replace();
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        return id;
    }

    function removeMessage(id) {
        const msg = document.getElementById(id);
        if (msg) msg.remove();
    }
});
