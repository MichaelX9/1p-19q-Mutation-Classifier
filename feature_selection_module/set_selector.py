### Generation fo test and train sets ###

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics'

randomized = np.random.permutation(159)
print(randomized)
training_set = []
test_set = []
for i in range(0,106):
    training_set.append(randomized[i]+1)
for i in range(106, 159):
    test_set.append(randomized[i]+1)
print(training_set)
print(test_set)
sorted_training = sorted(training_set)
sorted_test = sorted(test_set)
print(sorted_training)
print(sorted_test)

#training = [137, 15, 26, 22, 14, 102, 68, 79, 54, 17, 30, 65, 50, 115, 77, 134, 29, 6, 126, 11, 39, 81, 138, 155, 136, 48, 28, 86, 3, 145, 157, 73, 118, 150, 72, 131, 59, 98, 32, 156, 89, 7, 116, 76, 142, 4, 93, 129, 71, 80, 119, 132, 40, 51, 121, 46, 31, 140, 70, 18, 85, 19, 159, 99, 128, 66, 106, 117, 43, 8, 143, 103, 69, 90, 62, 60, 57, 61, 111, 139, 2, 75, 108, 58, 153, 25, 9, 88, 13, 83, 78, 101, 127, 33, 97, 92, 114, 55, 146, 20, 100, 141, 154, 5, 41, 1]
#sorted_training = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 17, 18, 19, 20, 22, 25, 26, 28, 29, 30, 31, 32, 33, 39, 40, 41, 43, 46, 48, 50, 51, 54, 55, 57, 58, 59, 60, 61, 62, 65, 66, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 83, 85, 86, 88, 89, 90, 92, 93, 97, 98, 99, 100, 101, 102, 103, 106, 108, 111, 114, 115, 116, 117, 118, 119, 121, 126, 127, 128, 129, 131, 132, 134, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 150, 153, 154, 155, 156, 157, 159]
#test = [53, 12, 151, 84, 135, 37, 63, 120, 94, 123, 152, 21, 67, 82, 130, 110, 105, 147, 35, 112, 34, 74, 10, 95, 133, 42, 38, 24, 56, 52, 47, 16, 149, 125, 107, 36, 96, 113, 45, 23, 109, 158, 122, 144, 148, 91, 64, 27, 87, 44, 124, 104]
#sorted_test = [10, 12, 16, 21, 23, 24, 27, 34, 35, 36, 37, 38, 42, 44, 45, 47, 52, 53, 56, 63, 64, 67, 74, 82, 84, 87, 91, 94, 95, 96, 104, 105, 107, 109, 110, 112, 113, 120, 122, 123, 124, 125, 130, 133, 135, 144, 147, 148, 149, 151, 152, 158]
csv_data = pd.read_csv(os.path.join(dataPath, 'TCIA_LGG_cases_159.csv'))
print(csv_data)
training_df = pd.DataFrame(columns=csv_data.columns)
test_df = pd.DataFrame(columns=csv_data.columns)
for i in range(0, len(sorted_training)):
    training_df.loc[i] = csv_data.loc[sorted_training[i] - 1]
for i in range(0, len(sorted_test)):
    test_df.loc[i] = csv_data.loc[sorted_test[i] - 1]
training_df.to_csv(os.path.join(dataPath, 'TrainingSet.csv'))
test_df.to_csv(os.path.join(dataPath, 'TestSet.csv'))