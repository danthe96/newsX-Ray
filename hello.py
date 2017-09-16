from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
import atexit
import cf_deployment_tracker
import os
import json
import nltk
import numpy as np
import pandas as pd

# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

db_name = 'mydb'
client = None
db = None

SIMILARITY_THRESHOLD = 0.8

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)

# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))


def clean_text(text):
    text = text.replace('<p>', '').replace('</p>', '').replace('<p/>', '') \
        .replace('\\n', ' ').replace('\\"', '\'').replace('\"', '\'')
    sentences = tokenizer.tokenize(data)
    return ''.join(['\"' + s + '\"\n' for s in sentences])


def calculate_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) * np.linalg.norm(vec2)


@app.route('/', methods=['POST'])
def check_similarity():
    reuters_text = clean_text(request.json['reuters'])
    news_text = clean_text(request.json['news'])

    reuters_sentences = reuters_data.index
    news_sentences = news_data.index

    info_in_both = []
    for n_s in news_sentences:
        for r_s in reuters_sentences:
            similarity = calculate_similarity(reuters_data.loc[r_s], news_data.loc[n_s])
            if similarity > SIMILARITY_THRESHOLD:
                info_in_both += {'news_sentence': n_s, 'original_sentence': r_s, 'score': similarity}

    info_in_news = [n_s for n_s in news_sentences if not any(x for x in info_in_both if x['news_sentence'] == n_s)]
    info_in_reuters = [r_s for r_s in reuters_sentences if not any(x for x in info_in_both if x['reuters_sentence'] == r_s)]

    return jsonify({
        'matched_sentences': info_in_both,
        'news_additions': info_in_news,
        'omitted_sentences': info_in_reuters
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
