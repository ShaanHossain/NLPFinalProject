import pandas

td = pandas.read_pickle('./datasets/t-davidson-hate-speech/labeled_data.p')
# print(t_davidson)

td = td.iloc[:,[5,4]]
td_c02 = td.query('`class` == 0 or `class` == 2')
class0 = td.query('`class` == 0').iloc[:,[0]]
class2 = td.query('`class` == 2').iloc[:,[0]]
class0['class'] = [1 for _ in class0.index]
class2['class'] = [0 for _ in class2.index]
# print(class0)

# td_X = td_c02.iloc[:,0]
# td_Y = td_c02.iloc[:,1]

class0train = class0.sample(frac = .70, random_state = 34)
class0test = class0.drop(index=class0train.index)
class2train = class2.sample(frac = .70, random_state = 34)
class2test = class2.drop(index=class2train.index)

train = class0train.append(class2train)
test = class0test.append(class2test)

train.to_csv(path_or_buf="./datasets/t-davidson-hate-speech/train.csv", index=False)
test.to_csv(path_or_buf="./datasets/t-davidson-hate-speech/test.csv", index=False)