import pickle as pkl    
from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import mysql.connector as sql

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("login.html")

@app.route("/login",methods=["GET","POST"])
def login():
    con=sql.connect(user='root', password='Karthic@2206*',host='localhost',database='login')
    cur=con.cursor()

    username = request.form['username']
    password = request.form['password']
    cur.execute('select * from details')
    t=cur.fetchone()
    # Check if the username exists and the password matches
    if username ==t[0] and password==t[1]:
        return render_template("index.html")

    else:
        return render_template("login.html")



@app.route("/predict", methods=['get','post'])

def predict():
    team1 = str(request.args.get('team1'))
    team2 = str(request.args.get('team2'))

    toss_win = int(request.args.get('toss_winner'))
    choose = int(request.args.get('toss_decision'))

    with open('inv_vocab.pkl', 'rb') as f:
        inv_vocab = pkl.load(f)

    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)

    cteam1 = inv_vocab[team1]
    cteam2 = inv_vocab[team2]

    if cteam1 == cteam2:
        return redirect(url_for('index'))

    lst = np.array([cteam1, cteam2, choose, toss_win], dtype='int32').reshape(1,-1)

    prediction = model.predict(lst)

    if prediction == 0:
        return render_template('success.html', data=team1)


    else:
        return render_template('success.html', data=team2)

if __name__=='__main__':
    app.run(host='localhost',port=5000,debug=True)
