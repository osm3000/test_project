from distutils.log import debug
import json
from flask import Flask, request, jsonify
import sentense_embedding
import thought_classification

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello Limbic Machine Learning Test!\n'


@app.route('/get_classification', methods=['GET'])
def classify_sentense():
    sentense = request.args.get('data')
    quantized_model = request.args.get('quantized_model')

    quantized_model = True if quantized_model == 'true' else False

    print(quantized_model, type(quantized_model))
    text_embedding = sentense_embedding.return_embedding(sentenses=[sentense])
    classification_results = thought_classification.perform_classification(
        model_type=quantized_model,
        embedded_sentences=text_embedding)
    # return f"Repeating the data: {sentense} :P \n {text_embedding}\nSentences Classification:{classification_results}\n"
    # return f"Repeating the data: {sentense} --- Sentences Classification:{classification_results}\n"
    return {
        'data_payload': {
            'original_sentence': sentense,
            'quantized_model': quantized_model,
            'thought_classification': classification_results    
        }
    }



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)