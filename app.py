import numpy as np
import pickle
import os
from preprocess import preprocess
import sqlite3
from flask import Flask,render_template,request,jsonify
import tensorflow as tf
from keras.models import load_model,model_from_json
from keras.layers import TextVectorization

logisticClassifier=pickle.load(open(os.path.join('pickleObjects','logisticClassifier.pkl'),'rb'))



##                            For loading text vectorization layer

# loaded_vectorizer=pickle.load(open(os.path.join('pickleObjects','vectorizer.pkl'),'rb'))
# new_vectorizer = TextVectorization.from_config(loaded_vectorizer['config'])
# new_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
# new_vectorizer.set_weights(loaded_vectorizer['weights'])

##                              For loading lstm model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# load weights into new model
# model.load_weights("model.h5")

#model=load_model('IMDBReviewsLSTMClassifier.h5')

db_path=os.path.join('reviews_database.sqlite')



# Initial Creation of reviews database to store reviews submitted by users
# conn=sqlite3.connect('reviews_database.sqlite')
# c=conn.cursor()
# c.execute("CREATE TABLE reviews_table(reviews TEXT,sentiment INTEGER, date TEXT)")
# review1='The movie was superb'
# review2='Not a nice movie'
# c.execute("INSERT INTO reviews_table VALUES(?,?,DATETIME('now'))",(review1,1))
# c.execute("INSERT INTO reviews_table VALUES(?,?,DATETIME('now'))",(review2,1))
# conn.commit()
# conn.close()

def classify(text):
    labels={0:'Negative',1:'Positive'}
    label=labels[logisticClassifier.predict(([preprocess(text)]))[0]]
    probability=round(np.max(logisticClassifier.predict_proba([preprocess(text)]))*100,2)
    return label,probability

def sqlite_entry(path,text,sentiment):
    conn=sqlite3.connect(path)
    c=conn.cursor()
    c.execute("INSERT INTO reviews_table VALUES(?,?,DATETIME('now'))",(text,sentiment))
    conn.commit()
    conn.close()

##For updating weights of lstm model for new reviews
def train(text,sentiment):
    x=new_vectorizer([preprocess(text)]).numpy()
    y=np.asarray([sentiment])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x,y)
    model.save_weights('model.h5')

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('reviewForm.html')

@app.route('/results',methods=['GET','POST'])
def results():
    if request.method=='POST':
        review=request.form['review']
        y,prob=classify(review)
        print(review,' ',y)
        return render_template('results.html',content=review,prediction=y,probability=prob)
    
@app.route('/thanks',methods=['GET','POST'])
def feedback():
    if request.method=='POST':
        feedback=request.form['feedbackButton']
        review=request.form['content']
        sentiment=request.form['prediction']
        labels={'Negative':0,'Positive':1}
        y=labels[sentiment]
        if feedback=='Incorrect prediction':
            y=int(not(y))
        sqlite_entry(db_path,review,y)
        return render_template('thanks.html')
    
if __name__=='__main__':
    app.run()