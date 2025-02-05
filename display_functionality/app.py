from flask import Flask, render_template

app = Flask(__name__)

def get_data():
    return {
        "väder": "☀️",  # Exempel på emoji för soligt väder
        "avstånd": 150,
        "antal_nu": 3,
        "antal_totalt": 10,
        "extra": ""  # Tom till att börja med
    }

@app.route("/")
def index():
    data = get_data()
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)