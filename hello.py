from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
import atexit
import cf_deployment_tracker
import os
import json
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import subprocess

# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

db_name = 'mydb'
client = None
db = None

SIMILARITY_THRESHOLD = 0.88

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
stop = set(stopwords.words('english'))


def clean_text(text):
    original_mapping = {}
    text = text.replace('<p>', '').replace('</p>', '').replace('<p/>', '')
    sentences = tokenizer.tokenize(data)
    basic_sentences = []
    for sentence in sentences:
        sentence2 = sentence.replace('\\n', ' ').replace('\\"', '\'').replace('\"', '\'')
        sentence2 = ' '.join([i for i in line.lower().split() if i not in stop])
        basic_sentences.append(sentence2)
        original_mapping[sentence2] = sentence

    return (''.join(['\"' + s + '\"\n' for s in basic_sentences]), original_mapping)


def calculate_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) * np.linalg.norm(vec2)


@app.route('/', methods=['POST'])
def check_similarity():
    reuters_text, original_r_s = clean_text(request.json['reuters'])
    news_text, original_n_s = clean_text(request.json['news'])

    reuters_id = request.json['reuters_id']
    newssource = request.json['source']

    if os.path.isfile(reuters_id + '_' + newssource + '_vec1.npy'):
        reuters_data = np.load(reuters_id + '_' + newssource + '_vec1.npy')
        news_data = np.load(reuters_id + '_' + newssource + '_vec2.npy')
    else:
        cutoff = len(reuters_text.split('\n'))
        f = open('text_temp', 'w')
        f.write(reuters_text)
        f.write(news_text)
        f.close()
        subprocess.Popen("fasttext/fasttext print-sentence-vectors fasttext/models/wiki.en.bin < text_temp > temp_result")

        result = pd.read_csv('temp_result', sep='\s+', quotechar='"', index_col=0, header=None)
        reuters_data = result.iloc[:(cutoff - 1)]
        news_data = result.iloc[cutoff:]
        np.save(reuters_id + '_' + newssource + '_vec1.npy', reuters_data)
        np.save(reuters_id + '_' + newssource + '_vec2.npy', news_data)

    reuters_sentences = reuters_data.index
    news_sentences = news_data.index

    info_in_both = []
    for n_s in news_sentences:
        for r_s in reuters_sentences:
            similarity = calculate_similarity(reuters_data.loc[r_s], news_data.loc[n_s])
            if similarity > SIMILARITY_THRESHOLD:
                info_in_both += {'news_sentence': original_n_s[n_s], 'original_sentence': original_r_s[r_s], 'score': similarity}

    info_in_news = [original_n_s[n_s] for n_s in news_sentences if not any(x for x in info_in_both if x['news_sentence'] == n_s)]
    info_in_reuters = [original_r_s[r_s] for r_s in reuters_sentences if not any(x for x in info_in_both if x['reuters_sentence'] == r_s)]

    return jsonify({
        'matched_sentences': info_in_both,
        'news_additions': info_in_news,
        'omitted_sentences': info_in_reuters
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
