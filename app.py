import csv
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, RepeatVector
import math

def read(filename,n):
    path = './data/' + filename + '.csv'
    with open(path,newline='',encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile)
        result = list()
        for row in rows:
            result.append([row[0],row[2],row[3],row[4]])
     
    return result[n:]



def readinput():
    M1,M2,M3,M4,M5,M6=2019,12,31,37351,4398,16
    path = './data/input.csv'
    with open(path,newline = '') as csvfile:
        rows = csv.reader(csvfile)
        result = list()
        for i in rows:
            temp = [float(i[0][0:4])/M1,float(i[0][4:6])/M2,float(i[0][6:])/M3,float(i[1])/M4,float(i[2])/M5,float(i[3])/M6]
            result.append(temp)
    result = np.array(result).reshape((1,7,6))        
    return result

def writecsv(result):
    with open('./submission.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date','peak_load(MW)'])
        for i in range(7):
            temp1 = str(20190402+i)
            temp2 = int(result[0][i+1])
            writer.writerow([temp1,temp2])
    

def build_model(seq_len):
    model = Sequential()
    model.add(LSTM(20, return_sequences=False))
    model.add(Dense(20, activation="relu"))
    model.add(RepeatVector(seq_len))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(1, return_sequences=True))
    model.compile(loss="mse", optimizer='adam')

    return model

def buildseq(data,m,n):
    k = len(data)-(m+n-1)
    resultx = []
    resulty = []
    for i in range(k):
        temp1 = data[i:i+m]
        temp2 = []      
        for j in range(n):
            temp2.append([data[i+m+j][3]]) 
        resultx.append(temp1)
        resulty.append(temp2)
    return resultx,resulty

def main():
    #兩筆資料時間不連續
    d1 = read('2017',1)
    d2 = read('台灣電力公司_過去電力供需資訊',335)

    M1,M2,M3,M4,M5,M6=2019,12,31,37351,4398,16
    cd = list()

    for i in d1 :
        temp = [float(i[0][0:4])/M1,float(i[0][4:6])/M2,float(i[0][6:])/M3,float(i[1])/M4,float(i[2])/M5,float(i[3])/M6]
        cd.append(temp)
        
    for i in d2 :
        temp = [float(i[0][0:4])/M1,float(i[0][4:6])/M2,float(i[0][6:])/M3,float(i[1])/M4,float(i[2])/M5,float(i[3])/M6]
        cd.append(temp)
        

    #建立序列
    x,y = buildseq(cd,7,8)
    X = np.array(x)
    Y = np.array(y)
   
    model = build_model(8)
    model.fit(X,Y,epochs=200, batch_size=100)

    #預測輸入
    inp = readinput()
    pre = model.predict(inp)*M4
    writecsv(pre)
    
if __name__ == '__main__':
    main()