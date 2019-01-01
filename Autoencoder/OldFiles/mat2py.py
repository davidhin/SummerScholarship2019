import pickle
with open('a.pkl', 'rb') as fin :
    a = pickle.load(fin) 
    print(a)
