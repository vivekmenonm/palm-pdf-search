from flask import Flask, request
import pandas as pd
from vertexai.preview.language_models import TextGenerationModel

app = Flask(__name__)

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

@app.route('/generate-answer', methods=['POST'])
def generate_answer():
    context = request.form.get('context')
    question = request.form.get('question')

    prompt = f"""Answer the question given in the context below:
    Context: {context}?\n
    Question: {question} \n
    Answer:
    """

    response = generation_model.predict(prompt).text
    print("The response is:", response)
    return response

@app.route('/generate-response', methods=['POST'])
def generate_summary():
    prompt = request.form.get('prompt')

    response = generation_model.predict(
        prompt, temperature=0.2, max_output_tokens=1024, top_k=40, top_p=0.8
    ).text

    return response

if __name__ == '__main__':
    app.run(debug=True)
