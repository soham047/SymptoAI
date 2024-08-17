from flask import Flask, render_template, request
import pickle
import sklearn
import numpy as np

model = pickle.load(open('lr1','rb'))
sc = pickle.load(open('scaler.pkl','rb'))
label = pickle.load(open('label.pkl','rb'))

app =Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    
    return render_template("index.html")
@app.route("/submit", methods=["POST","GET"])
def submit():
    data1 = request.form['i1']
    data2 = request.form['i2']
    data3 = request.form['i3']
    data4 = request.form['i4']
    data5 = request.form['i5']
    data6 = request.form['i6']
    data7 = request.form['i7']
    data8 = request.form['i8']
    data9 = request.form['i9']
    data10 = request.form['i10']
    data11 = request.form['i11']
    data12 = request.form['i12']
    data13 = request.form['i13']
    data14 = request.form['i14']    
    data15 = request.form['i15']
    data16 = request.form['i16']
    data17 = request.form['i17']
    data18 = request.form['i18']
    data19 = request.form['i19']
    data20 = request.form['i20']
    data21 = request.form['i21']
    data22 = request.form['i22']
    data23 = request.form['i23']
    data24 = request.form['i24']
    d = [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24]
    
    for i in range(len(d)):
        try:
            if d[i] == '':
                d[i] = float(0)
            else:
                d[i]   = float(d[i])
        except:
            return f"Invalid input data format. Only enter Numeric Data. Return to the previous page."
            
    d = np.array(d).reshape(1,-1)
    d = sc.transform(d)
    a=model.predict(d)
    a= label.inverse_transform(a)
    a=''.join(map(str,a))
    
    return render_template('index.html', status=a) 
if __name__ == "__main__":
    app.run(debug='True')
    
    
    
    
    