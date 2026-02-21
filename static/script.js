// change this in script.js
const API_BASE_URL = '/api';

let currentVideoUrl = null;
let isProcessing = false;

// DOM elements
const videoUrlInput = document.getElementById('videoUrlInput');
const processBtn = document.getElementById('processBtn');
const transcriptInput = document.getElementById('transcriptInput');
const processTranscriptBtn = document.getElementById('processTranscriptBtn');
const videoStatus = document.getElementById('videoStatus');
const videoInputSection = document.getElementById('videoInputSection');
const chatContainer = document.getElementById('chatContainer');
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const newVideoBtn = document.getElementById('newVideoBtn');

// Event listeners
processBtn.addEventListener('click', processVideo);
if (processTranscriptBtn) {
    processTranscriptBtn.addEventListener('click', processTranscript);
}
sendBtn.addEventListener('click', sendQuestion);
newVideoBtn.addEventListener('click', resetToVideoInput);

questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

async function processVideo() {
    const url = videoUrlInput.value.trim();

    if (!url) {
        showStatus('Please enter a YouTube video URL', 'error');
        return;
    }

    if (!isValidYouTubeUrl(url)) {
        showStatus('Please enter a valid YouTube URL', 'error');
        return;
    }

    isProcessing = true;
    processBtn.disabled = true;
    showStatus('Processing video... This may take a moment.', 'loading');

    try {
        const response = await fetch(`${API_BASE_URL}/process-video`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url }),
        });

        const data = await response.json();

        if (response.ok) {
            currentVideoUrl = url;
            showStatus(data.message, 'success');
            setTimeout(() => {
                showChatInterface();
            }, 1000);
        } else {
            showStatus(data.error || 'Failed to process video', 'error');
        }
    } catch (error) {
        showStatus('Error connecting to server. Make sure the backend is running.', 'error');
        console.error('Error:', error);
    } finally {
        isProcessing = false;
        processBtn.disabled = false;
    }
}

async function processTranscript() {
    const url = videoUrlInput.value.trim();
    const transcript = transcriptInput ? transcriptInput.value.trim() : '';

    if (!transcript) {
        showStatus('Please paste a transcript before using this option.', 'error');
        return;
    }

    isProcessing = true;
    processBtn.disabled = true;
    if (processTranscriptBtn) {
        processTranscriptBtn.disabled = true;
    }
    showStatus('Processing pasted transcript... This may take a moment.', 'loading');

    try {
        // Use the provided URL as the key if present, otherwise let the backend generate one
        const body = url ? { transcript, video_url: url } : { transcript };

        const response = await fetch(`${API_BASE_URL}/process-transcript`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        if (response.ok) {
            currentVideoUrl = data.video_url;
            showStatus(data.message, 'success');
            setTimeout(() => {
                showChatInterface();
            }, 1000);
        } else {
            showStatus(data.error || 'Failed to process transcript', 'error');
        }
    } catch (error) {
        showStatus('Error connecting to server. Make sure the backend is running.', 'error');
        console.error('Error:', error);
    } finally {
        isProcessing = false;
        processBtn.disabled = false;
        if (processTranscriptBtn) {
            processTranscriptBtn.disabled = false;
        }
    }
}

async function sendQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    if (!currentVideoUrl) {
        showStatus('Please process a video first', 'error');
        return;
    }

    // Add user message to chat
    addMessage(question, 'user');
    questionInput.value = '';
    sendBtn.disabled = true;

    // Show loading indicator
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API_BASE_URL}/ask-question`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                video_url: currentVideoUrl
            }),
        });

        const data = await response.json();

        // Remove loading indicator
        removeMessage(loadingId);

        if (response.ok) {
            addMessage(data.answer, 'bot');
        } else {
            addMessage(data.error || 'Failed to get answer', 'bot');
        }
    } catch (error) {
        removeMessage(loadingId);
        addMessage('Error connecting to server. Please try again.', 'bot');
        console.error('Error:', error);
    } finally {
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message loading';
    messageDiv.id = 'loading-' + Date.now();

    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';

    messageDiv.appendChild(typingDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageDiv.id;
}

function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}

function showStatus(message, type) {
    videoStatus.textContent = message;
    videoStatus.className = `status-message ${type}`;
}

function showChatInterface() {
    videoInputSection.style.display = 'none';
    chatContainer.style.display = 'flex';
    questionInput.focus();

    // Add welcome message
    addMessage('Video processed successfully! Ask me anything about this video.', 'bot');
}

function resetToVideoInput() {
    currentVideoUrl = null;
    chatMessages.innerHTML = '';
    chatContainer.style.display = 'none';
    videoInputSection.style.display = 'block';
    videoUrlInput.value = '';
    if (transcriptInput) {
        transcriptInput.value = '';
    }
    videoStatus.textContent = '';
    videoStatus.className = 'status-message';
}

function isValidYouTubeUrl(url) {
    const patterns = [
        /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\/.+/,
        /^https?:\/\/youtu\.be\/.+/,
    ];
    return patterns.some(pattern => pattern.test(url));
}

// Check if backend is running on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            showStatus('Backend server is not responding. Please make sure the Flask server is running.', 'error');
        }
    } catch (error) {
        showStatus('Cannot connect to backend server. Please start the Flask server (python app.py)', 'error');
    }
});

