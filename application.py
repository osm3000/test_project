from distutils.log import debug
import json
from time import time
from flask import Flask, request, jsonify
import sentense_embedding
import thought_classification
import time
import datetime

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello Limbic Machine Learning Test!\n'


@app.route('/get_classification', methods=['GET'])
def classify_sentense():
    start_time = time.time()
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

    end_time = time.time() - start_time

    ip_addr = request.remote_addr
    with open('logs.log', 'a') as file_handle:
        print(f'Time: {datetime.datetime.now()} - IP: {ip_addr}',
              file=file_handle)

    return {
        'data_payload': {
            'original_sentence': sentense,
            'quantized_model': quantized_model,
            'thought_classification': classification_results,
            'processing_time_seconds': end_time,
            'your_ip': ip_addr
        }
    }



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)