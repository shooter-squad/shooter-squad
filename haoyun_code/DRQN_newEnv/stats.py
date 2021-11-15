import numpy as np
import matplotlib.pyplot as plt
import sys

file = open("stats_lstm2.txt","r")

score, avg, best, eps, step = [],[],[],[],[]
line = file.readline()
while line:
    tmp = line.split(":")
    if("," in tmp[1]):
        score.append(float(tmp[2][1:-15]))
        avg.append(float(tmp[3][1:-12]))
        best.append(float(tmp[4][1:-9]))
        eps.append(float(tmp[5][1:-7]))
        step.append(float(tmp[6][1:-1]))
    line = file.readline()

score = np.array(score)
avg = np.array(avg)
best = np.array(best)
eps = np.array(eps)
step = np.array(step)
index = np.arange(score.shape[0])


plt.scatter(index, avg, s= 1, label="average score")
plt.legend()
plt.savefig('stats2.jpg')
