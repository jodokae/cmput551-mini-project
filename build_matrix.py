import pymysql
import copy
import numpy as np
import time
import datetime

with open('query.sql', 'r') as queryFile:
    query=queryFile.read().replace('\n', ' ')
    
conn = pymysql.connect(host='localhost', user='archi', passwd='archi', db='archi')
cur = conn.cursor()
cur.execute(query)
dataSet = copy.copy(cur)

rows = len(cur.fetchall())

if rows == 0:
    exit

y = np.zeros(rows)
A = np.empty((rows,40))
A.fill(np.nan)

for rowIndex, row in enumerate(dataSet):
    if True:
        for i in range(len(row)):
            # lang case
            if i == 0:
                #print(row[i])
                if row[i] == 'java':
                    A[rowIndex][0] = 1
                    A[rowIndex][1] = 0
                    A[rowIndex][2] = 0
                if row[i] == 'javascript':
                    A[rowIndex][0] = 0
                    A[rowIndex][1] = 1
                    A[rowIndex][2] = 0
                if row[i] == 'ruby':
                    A[rowIndex][0] = 0
                    A[rowIndex][1] = 0
                    A[rowIndex][2] = 1
            # num_commits in push
            elif i == 1:
                A[rowIndex][3] = row[i]
            # prev commit res
            elif i == 2:
                if row[i] == 'build_found':
                    A[rowIndex][4] = 1
                    A[rowIndex][5] = 0
                    A[rowIndex][6] = 0
                if row[i] == 'merge_found':
                    A[rowIndex][4] = 0
                    A[rowIndex][5] = 1
                    A[rowIndex][6] = 0
                if row[i] == 'no_previous_build':
                    A[rowIndex][4] = 0
                    A[rowIndex][5] = 0
                    A[rowIndex][6] = 1
            # tr status
            elif i == 36:
                if row[i] == 'passed':
                    y[rowIndex] = 1
                else:
                    y[rowIndex] = 0
            #dates
            elif i == 25 or i == 26:
                if row[i] != None:
                    A[rowIndex][i+4] = 0
                    #A[rowIndex][i+4] = time.mktime(row[i].timetuple())
            else:
                if row[i] != None and row[i] != '':
                    #print(row[i])
                    A[rowIndex][i+4] = float(row[i])

np.save('A.npy', A)
np.save('y.npy', y)
