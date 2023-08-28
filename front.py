from flask import Flask, render_template,request
from last import evaluate,encoder1,decoder1



app = Flask(__name__)

def evaluateRandomely(encoder, decoder,input_text):
    output_words, attentions = evaluate(encoder, decoder, input_text)
    output_sentence = ' '.join(output_words)
    return output_sentence

@app.route('/')
def translator():
    return render_template('index.html')

@app.route('/try.html')
def home():
    return render_template('try.html')

@app.route("/last.py", methods=["POST"])
def before():
    if request.method == "POST":
        input_text = request.form['english-input']
        print("Input Text:", input_text)
        output = evaluateRandomely(encoder1, decoder1,input_text)
    else:
        return render_template("try.html")
    return render_template('try.html', output=output, input_text=input_text)

if __name__ == '__main__':
   app.run(debug=True)