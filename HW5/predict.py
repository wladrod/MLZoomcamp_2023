from flask import Flask
from flask import request
from flask import jsonify
import pickle

file_model = 'model2.bin'
file_dv = 'dv.bin'
with open(file_model, 'rb') as f_in:
    model = pickle.load(f_in)
    
with open(file_dv, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('credit')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1].round(3)
    result = {
        'credit probability': float(y_pred)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)