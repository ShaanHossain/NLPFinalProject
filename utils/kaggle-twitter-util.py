import pandas

td = pandas.read_csv("./datasets/kaggle-twitter/train.csv", index_col=0, header = 0, names=["class","tweet"])

class0 = td.query('`class` == 0')
class1 = td.query('`class` == 1')

class0 = class0.iloc[:,[1,0]]
class1 = class1.iloc[:,[1,0]]

class0train = class0.sample(frac = .70, random_state = 34)
class0test = class0.drop(index=class0train.index)
class1train = class1.sample(frac = .70, random_state = 34)
class1test = class1.drop(index=class1train.index)



train = class0train.append(class1train)
test = class0test.append(class1test)

# print(train)
# print(test)

train.to_csv(path_or_buf="./datasets/kaggle-twitter/newtrain.csv", index=False)
test.to_csv(path_or_buf="./datasets/kaggle-twitter/newtest.csv", index=False)