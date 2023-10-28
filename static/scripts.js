function sendMessage1(){
    const message = document.getElementById('message-input').value;   
    if (message.trim() !== '') {
        sendMessage(message);
        document.getElementById('message-input').value = '';
    }
}

function appendMessage(content, isBot) {
    const chatbox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.className = isBot ? 'chat-bot' : 'chat-user';
    messageDiv.innerHTML = `<div class="chat-message-content">${content}</div>`;
    chatbox.appendChild(messageDiv);
}

function sendMessage(message) {
    
    appendMessage(message, false);
    const xhr = new XMLHttpRequest();
        xhr.open('POST', '/chatbot', true);
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200) {
                    const result = JSON.parse(xhr.responseText);
                    appendMessage(result.response, true);
                } else {
                    alert('Error');
                }
            }
        };
        xhr.send(`question=${encodeURIComponent(message)}`);
}
