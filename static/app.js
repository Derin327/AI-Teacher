const chatHistory = document.getElementById('chatHistory');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const uploadStatus = document.getElementById('uploadStatus');

async function handleUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (file.type !== "application/pdf") {
        alert("Please upload a valid PDF file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    uploadStatus.classList.remove('hidden');
    uploadStatus.textContent = "Uploading & indexing...";
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (response.ok) {
            uploadStatus.textContent = "✅ Indexed successfully!";
            uploadStatus.classList.add("text-green-600");
            uploadStatus.classList.remove("text-gray-500");
            
            // Append system message
            appendMessage("🤖", `I've successfully read **${file.name}**. What would you like to know about it?`, 'system');
        } else {
            throw new Error(result.detail || "Upload failed");
        }
    } catch (error) {
        console.error(error);
        uploadStatus.textContent = "❌ Failed";
        uploadStatus.classList.add("text-red-600");
        uploadStatus.classList.remove("text-gray-500");
        alert("Error: " + error.message);
    }
    
    setTimeout(() => {
        uploadStatus.classList.add('hidden');
        uploadStatus.className = "text-sm hidden"; // Reset classes
        event.target.value = ""; // Reset file input
    }, 5000);
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;
    
    messageInput.value = "";
    messageInput.disabled = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = `<svg class="animate-spin h-5 w-5 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`;

    // Add user message to UI
    appendMessage("🧑", text, 'user');
    
    // Create placeholder for AI response
    const botMsgId = appendMessage("🤖", `<span class="animate-pulse">Thinking...</span>`, 'bot', true);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            updateBotMessage(botMsgId, data);
        } else {
            throw new Error(data.detail || "Failed to fetch response");
        }
    } catch (error) {
        console.error(error);
        updateBotMessageText(botMsgId, `**Error:** ${error.message}`);
    } finally {
        messageInput.disabled = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = "Send";
        messageInput.focus();
        scrollToBottom();
    }
}

function appendMessage(icon, text, role, isHtml = false) {
    const id = "msg_" + Math.random().toString(36).substr(2, 9);
    
    const wrapper = document.createElement("div");
    wrapper.className = "flex gap-4";
    wrapper.id = id;
    
    const iconDiv = document.createElement("div");
    iconDiv.className = `w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 text-xl ${role === 'user' ? 'bg-indigo-100' : 'bg-blue-100'}`;
    iconDiv.textContent = icon;
    
    const contentDiv = document.createElement("div");
    contentDiv.className = `flex-1 rounded-2xl p-4 text-gray-800 ${role === 'user' ? 'bg-indigo-50 border border-indigo-100' : 'bg-gray-50 border border-gray-200'}`;
    
    if (isHtml) {
        contentDiv.innerHTML = text;
    } else {
        contentDiv.innerHTML = `<div class="markdown-body">${marked.parse(text)}</div>`;
    }
    
    wrapper.appendChild(iconDiv);
    wrapper.appendChild(contentDiv);
    
    chatHistory.appendChild(wrapper);
    scrollToBottom();
    
    return id;
}

function updateBotMessageText(id, text) {
    const wrapper = document.getElementById(id);
    if (!wrapper) return;
    const contentDiv = wrapper.children[1];
    contentDiv.innerHTML = `<div class="markdown-body text-red-500">${marked.parse(text)}</div>`;
}

function updateBotMessage(id, data) {
    const wrapper = document.getElementById(id);
    if (!wrapper) return;
    
    const contentDiv = wrapper.children[1];
    contentDiv.innerHTML = ""; // Clear "Thinking..."
    
    // Add text answer
    const textContainer = document.createElement("div");
    textContainer.className = "markdown-body text-sm md:text-base";
    textContainer.innerHTML = marked.parse(data.answer || "No answer generated.");
    contentDiv.appendChild(textContainer);
    
    // Add media container if audio or video is present
    if (data.audio_url || data.video_url) {
        const mediaContainer = document.createElement("div");
        mediaContainer.className = "mt-4 flex flex-col gap-3 pt-4 border-t border-gray-200";
        
        if (data.video_url) {
            const videoLabel = document.createElement("span");
            videoLabel.className = "text-xs font-semibold text-gray-500 uppercase";
            videoLabel.textContent = "Teacher Video";
            
            const videoEl = document.createElement("video");
            videoEl.controls = true;
            videoEl.className = "w-full max-w-sm rounded-lg shadow-sm border border-gray-200";
            videoEl.src = data.video_url;
            
            mediaContainer.appendChild(videoLabel);
            mediaContainer.appendChild(videoEl);
        } else if (data.audio_url) {
            const audioLabel = document.createElement("span");
            audioLabel.className = "text-xs font-semibold text-gray-500 uppercase";
            audioLabel.textContent = "Audio Response";
            
            const audioEl = document.createElement("audio");
            audioEl.controls = true;
            audioEl.className = "w-full outline-none";
            audioEl.src = data.audio_url;
            
            mediaContainer.appendChild(audioLabel);
            mediaContainer.appendChild(audioEl);
        }
        
        contentDiv.appendChild(mediaContainer);
    }
    
    scrollToBottom();
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
