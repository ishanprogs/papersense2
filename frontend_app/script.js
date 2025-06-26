document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading');
    const chatWindow = document.querySelector('.chat-window');

    // Upload elements
    const fileUploadInput = document.getElementById('file-upload-input');
    const uploadButton = document.getElementById('upload-button');
    const uploadStatusDiv = document.getElementById('upload-status');

    // --- VERY IMPORTANT: Replace with your deployed Function URLs ---
    const askFunctionApiUrl = 'https://pdf-rag-function-new.azurewebsites.net/api/AskQuestion?';
    const uploadFunctionApiUrl = 'https://pdf-rag-function-new.azurewebsites.net/api/upload?code=6zs319je6IvChwIdtxjCvYC4XlKKd_dd1p3sLBCACcXdAzFuejxGxQ==';

    // Auto-resize textarea based on content
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        let scrollHeight = userInput.scrollHeight;
        userInput.style.height = Math.min(scrollHeight, 150) + 'px';
    });

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    // --- Enhanced Upload Functionality ---
    if (uploadButton) {
        uploadButton.addEventListener('click', async () => {
            if (!fileUploadInput.files || fileUploadInput.files.length === 0) {
                setUploadStatus("Please select a PDF, JSON, TXT, or CSV file to upload.", true);
                return;
            }

            if (!uploadFunctionApiUrl || uploadFunctionApiUrl === '<YOUR_AZURE_FUNCTION_APP_UPLOADPDF_URL>') {
                setUploadStatus("ERROR: The Upload Document Function URL is not configured in script.js.", true);
                return;
            }

            const file = fileUploadInput.files[0];
            
            // ADDED: Client-side file type validation
            const allowedExtensions = ['.pdf', '.json', '.txt', '.csv'];
            const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
            if (!allowedExtensions.includes(fileExtension)) {
                setUploadStatus(`Invalid file type. Please select: ${allowedExtensions.join(', ')} files only.`, true);
                return;
            }
            
            // ADDED: File size validation (20MB limit to match backend)
            const maxSizeMB = 20;
            const fileSizeMB = file.size / (1024 * 1024);
            if (fileSizeMB > maxSizeMB) {
                setUploadStatus(`File size (${fileSizeMB.toFixed(2)} MB) exceeds the ${maxSizeMB} MB limit.`, true);
                return;
            }
            const formData = new FormData();
            formData.append('file', file);

            // ENHANCED: Better status messages
            setUploadStatus(`Uploading ${file.name} (${fileSizeMB.toFixed(2)} MB)...`, false);
            uploadButton.disabled = true;
            fileUploadInput.disabled = true;
            try {
                const response = await fetch(uploadFunctionApiUrl, {
                    method: 'POST',
                    body: formData,
                });
                const responseText = await response.text();
                if (!response.ok) {
                    let errorDetail = responseText;
                    try {
                        const errJson = JSON.parse(responseText);
                        errorDetail = errJson.error || errJson.message || responseText;
                    } catch (e) { /* ignore if not json */ }
                    throw new Error(`Upload failed: ${response.status} - ${errorDetail}`);
                }
                setUploadStatus(`✅ ${file.name} uploaded successfully! Processing ${fileExtension.toUpperCase()} file...`, false, true);
                fileUploadInput.value = '';
            } catch (error) {
                console.error("Upload error:", error);
                setUploadStatus(`❌ Error: ${error.message}`, true);
            } finally {
                uploadButton.disabled = false;
                fileUploadInput.disabled = false;
            }
        });
    }

    function setUploadStatus(message, isError = false, isSuccess = false) {
        uploadStatusDiv.textContent = message;
        uploadStatusDiv.className = 'upload-status';
        if (isError) {
            uploadStatusDiv.classList.add('error');
        } else if (isSuccess) {
            uploadStatusDiv.classList.add('success');
        }
        if (isError || isSuccess) {
            setTimeout(() => {
                if (uploadStatusDiv.textContent === message) {
                    uploadStatusDiv.textContent = '';
                    uploadStatusDiv.className = 'upload-status';
                }
            }, 7000);
        }
    }

    function scrollToBottom() {
        setTimeout(() => {
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }, 0);
    }

    function addMessage(sender, text, sources = [], metadata = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');

        // Basic Markdown-like formatting
        contentDiv.innerHTML = text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
        messageDiv.appendChild(contentDiv);

        // Enhanced sources display
        if (sender === 'assistant' && sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.classList.add('sources');
            sourcesDiv.innerHTML = `<strong>Sources:</strong> ${sources.join(', ')}`;
            contentDiv.appendChild(sourcesDiv);
        }
        if (sender === 'assistant' && metadata) {
            const metaDiv = document.createElement('div');
            metaDiv.classList.add('metadata-debug');
            metaDiv.innerHTML = `<small>Strategy: ${metadata.strategy_used || 'N/A'}. Reasoning: ${metadata.llm_detection_reasoning || 'N/A'}</small>`;
            contentDiv.appendChild(metaDiv);
        }
        chatLog.appendChild(messageDiv);
        scrollToBottom();
    }

    async function sendMessage() {
        const question = userInput.value.trim();
        if (!question) return;
        // Get selected query mode from radio buttons
        const selectedQueryModeInput = document.querySelector('input[name="queryMode"]:checked');
        const queryMode = selectedQueryModeInput ? selectedQueryModeInput.value : 'document_search_generic';
        if (!askFunctionApiUrl || askFunctionApiUrl.includes('<YOUR_AZURE_FUNCTION_APP_ASKQUESTION_URL>')) {
            addMessage('assistant', 'ERROR: The Ask Question Function URL is not configured in script.js.');
            return;
        }
        addMessage('user', question);
        userInput.value = '';
        userInput.style.height = 'auto';
        loadingIndicator.classList.remove('hidden');
        sendButton.disabled = true;
        userInput.disabled = true;
        scrollToBottom();
        try {
            const response = await fetch(askFunctionApiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    question: question, 
                    query_mode: queryMode 
                }),
            });
            const responseData = await response.json();
            if (!response.ok || responseData.error) {
                const errorText = responseData?.error || responseData?.answer || `Request failed with status ${response.status}`;
                throw new Error(errorText);
            }
            addMessage('assistant', responseData.answer || "No answer received.", responseData.sources, responseData.metadata);
        } catch (error) {
            console.error("Error fetching response:", error);
            let displayError = error.message;
            if (error.message.toLowerCase().includes("failed to fetch")) {
                displayError = "Network error: Could not connect to the Q&A service. Please check your connection or the service status.";
            } else if (error.message.toLowerCase().includes("unexpected token")) {
                displayError = "Received an invalid response from the server. Please try again.";
            }
            addMessage('assistant', `Sorry, I encountered an error: ${displayError}`);
        } finally {
            loadingIndicator.classList.add('hidden');
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    userInput.focus();
});
