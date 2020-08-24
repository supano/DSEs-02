import flask
from flask import request as fl_requests, jsonify
from flask_cors import CORS, cross_origin
import predict as pd 

app = flask.Flask(__name__, static_url_path='/static')
cors = CORS(app)

@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')

@app.route('/api', methods=['GET'])
@cross_origin()
def api():
    if 'url' in fl_requests.args:
        url = fl_requests.args['url']
    else:
        return "Error: No url field provided. Please specify an url."

    return pd.predictimg(url)

if __name__ == "__main__":
    app.run()