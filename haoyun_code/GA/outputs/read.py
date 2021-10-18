import numpy as np
import sys


data = np.load('output3/-10-17-2021_19-08_NN=RNNIndividual_POPSIZE=20_GEN=500_PMUTATION_0.1507999999999986_PCROSSOVER_0.8_I=246_SCORE=140.0.npy')

print(data.shape)
print(data)


# output = open('output2.txt','w+')

# index = 1
# for i in data:
#     output.write("generation: ")
#     output.write(str(index))
#     output.write(", score: ")
#     output.write(str(i))
#     output.write("\n")
#     index+=1

# output.close()