document.getElementById('send-btn').addEventListener('click', function() {
    const inputField = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    if (inputField.value.trim() !== '') {
        // 添加用户对话
        const userMessage = createChatBubble(inputField.value, 'user');
        chatBox.appendChild(userMessage);
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: inputField.value }),
        })
        .then(response => response.json())
        .then(data => {
            const chatBox = document.getElementById("chat-box");
            const assistantMessage = createChatBubble(data.reply, 'assistant');
            document.getElementById("user-input").value = "";
            chatBox.appendChild(assistantMessage);
            chatBox.scrollTop = chatBox.scrollHeight;
        });
    };
});

// 创建聊天气泡
function createChatBubble(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add(role);

    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('avatar');

    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add('chat-bubble');

    
    bubbleDiv.textContent = content;

    if (role === 'user') {
        messageDiv.appendChild(bubbleDiv);  // 气泡在前
        messageDiv.appendChild(avatarDiv);  // 头像在后
    } else if (role === 'assistant') {
        messageDiv.appendChild(avatarDiv);  // 头像在前
        messageDiv.appendChild(bubbleDiv);  // 气泡在后
    }
    return messageDiv;
};
