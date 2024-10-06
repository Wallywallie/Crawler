from flask import Flask, render_template, request, jsonify
import requests
import json
import os

app = Flask(__name__)

API_KEY = "sXu3ML0Q1VitaW9X98Zo2yJQ"
SECRET_KEY = "pYEw0w3kBLbBT8CDLqZR0NpoNbxmXzi4"

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    response = requests.post(url, params=params)
    return response.json().get("access_token")

def get_assistant_reply(user_message):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k?access_token=" + get_access_token()
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": user_message,
                "key": "1"
            }
        ],
        "temperature": 0.95,
        "top_p": 0.7,
        "penalty_score": 1,
        "collapsed": True
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=payload)
    result = response.json()
    print(result)
    assistant_message = result['result'] if 'result' in result and len(result['result']) > 1 else "Error retrieving response"
    return assistant_message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    assistant_reply = get_assistant_reply(user_message)
    return jsonify({"reply": assistant_reply})

if __name__ == '__main__':
    app.run(debug=True)
