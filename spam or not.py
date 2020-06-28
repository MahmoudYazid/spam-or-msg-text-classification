from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import keras as k


def predict_text():
    import numpy
    import sqlite3 as ql
    feature=[]
    msg=[]
    #
    con=ql.connect("text class.db")
    sql=con.cursor()
    sql.execute(""" SELECT msg,Category FROM train  LIMIT 82 """)
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer

    for add in sql.fetchall():
        feature.append('{}'.format(add[1]))



        msg.append('{}'.format(add[0]))

    vectorizer = CountVectorizer()
    vectorizer.fit(msg)

    x_train = vectorizer.transform(msg)

    convert_feature=LabelEncoder()
    y_train=convert_feature.fit_transform(feature)

    #_new_msg_array
    #_new_feature_array
    model=k.Sequential()
    model.add(k.layers.Dense(218,input_shape=(630,),activation='relu'))
    model.add(k.layers.Dropout(.7))
    model.add(k.layers.Dense(500, activation='relu'))
    model.add(k.layers.Dropout(.7))
    model.add(k.layers.Dense(1,activation='relu'))
    model.compile(optimizer='nadam',loss='mse')


    model.fit(x_train,y_train,epochs=1000,verbose=1)

    #


    predicted_array=[]
    sql.execute(""" SELECT msg FROM test_ LIMIT 90  """)
    for add_predicted in sql.fetchall():
        predicted_array.append(add_predicted[0])
    vectorizer = CountVectorizer()
    vectorizer.fit(predicted_array)
    x_predict = vectorizer.transform(predicted_array)
    final_array=model.predict(x_predict)
    ar_id=[]
    for identify in final_array[0:]:
        if identify==0:
            ar_id.append("ham")
        else:
            ar_id.append("spam")
    sql.execute(""" SELECT Category FROM test_ LIMIT 90  """)
    check_array=[]
    for check in sql.fetchall():
        check_array.append("{}".format(check[0]))
    true_or=[]
    # check
    for iter in range(85):
        if str(ar_id[iter])==str(check_array[iter]):
            true_or.append(1)
        else:
            true_or.append(0)
    print("true is ",true_or.count(1))
    print("false is ",true_or.count(0))
    print("acc :",(float(true_or.count(1)/float(true_or.count(1)+true_or.count(0)))))





predict_text()