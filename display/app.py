from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    data = {
        "väder": "☀️",  # Exempel på emoji
        "avstånd": 150,  # Exempel på meter
        "antal_nu": 42,
        "antal_totalt": 100,
        "extra": "Data"  # Placeholder för mittenboxen
    }
    return render_template("index.html", data=data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)