from flask import Flask, render_template, request, jsonify

from chatbot import ResponsibleChatbot

app = Flask(__name__)
chatbot = ResponsibleChatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    result = chatbot.chat(user_message)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
