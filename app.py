from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

app = Flask(__name__)


class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.pooler_output
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


model     = SentimentClassifier()
model.load_state_dict(torch.load('best_model.pth',map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('tokenizer/')


def predict(text):
    tokens = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    with torch.no_grad():
        outputs    = model(input_ids, attention_mask)
        probs       = F.softmax(outputs, dim=1)
        prediction  = torch.argmax(probs, dim=1).item()
        confidence  = probs[0][prediction].item() * 100 


    label = "Positive comment" if prediction == 1 else "Negative comment"
    return {'result': label, 'confidence': round(confidence, 1)}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    data = request.json
    text = data['text']
    result = predict(text)
    return jsonify({'result': result['result'], 'confidence': result['confidence']})

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)