import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, RepeatVector

def read(filename):
    path = './data/' + filename + '.csv'
    with open(path,newline='',encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile)
        count = 0
        print(type(rows))
        result = list()
        for row in rows:
            result.append([row[0],row[2],row[3],row[4]])
     
    return result

def build_model(seq_len, hidden_size):
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=False))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(RepeatVector(seq_len))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.compile(loss="mse", optimizer='adam')

    return model

def main():
    d = read('台灣電力公司_過去電力供需資訊')
    for i in d :
        print(i)

    model = build_model(7,10)
    

if __name__ == '__main__':
    main()