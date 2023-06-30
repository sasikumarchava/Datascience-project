from flask import Flask, render_template, request
from Admission_New import calc



app = Flask(__name__)# interface between my server and my application wsgi

import pickle
model = pickle.load(open(r'C:\Users\nagap\Downloads\University Admission Eligibility Predictor\University Admission Eligibility Predictor\model.pkl','rb'))


@app.route('/')#binds to an url
def helloworld():
    return render_template("index.html")

@app.route("/prediction", methods=["GET"])
def redirect_internal():
    return render_template("/prediction.html", code=302)

@app.route('/predicted', methods =['POST'])#binds to an url
def login():
    p =request.form["gs"]
    q= request.form["ts"]
    r= request.form["ur"]
    if (r=="1"):
        r_val=1
    if (r=="2"):
        r_val=2
    if (r=="3"):
        r_val=3
    if (r=="4"):
        r_val=4
    if (r=="5"):
        r_val=5
    s= request.form["sop"]
    t= request.form["lor"]
    u= request.form["cgpa"]
    v= request.form["rnd"]
    if (v=="1"):
        v_val=1
    if (v=="0"):
        v_val=0
    
    output = model.predict([calc(int(p),int(q),int(r_val),float(s),float(t),float(u),int(v_val))])
    print(output)  
        
    return render_template("prediction.html",submit_to_check_result = "The predicted chance is  " + str((output[0]*100))+" percentage" )

#@app.route('/admin')#binds to an url
#def admin():
   # return "Hey Admin How are you?"

if __name__ == '__main__' :
    app.run(debug= False)
    