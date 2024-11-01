import uuid
from flask import Flask, request, jsonify
from pmg_rag import RAGSystem
import pmg_db

app = Flask(__name__)
rag_sys = RAGSystem()


@app.route("/question", methods=["POST"])
def handle_question():
    data = request.json
    question = data["question"]

    if not question:
        return jsonify({"error": "No question provided"}), 400

    conversation_id = str(uuid.uuid4())

    answer_data = rag_sys.rag(question)

    result = {
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer_data["answer"],
    }

    pmg_db.save_conversation(
        conversation_id=conversation_id,
        question=question,
        answer_data=answer_data,
    )

    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    conversation_id = data["conversation_id"]
    feedback = data["feedback"]

    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400

    pmg_db.save_feedback(
        conversation_id=conversation_id,
        feedback=feedback,
    )

    result = {
        "message": f"Feedback received for conversation {conversation_id}: {feedback}"
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)