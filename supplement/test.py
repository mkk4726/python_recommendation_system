import pickle
with open('results/find_K_300.pickle', 'rb') as handle:
    print(pickle.load(handle)[0][0])