import pymysql
import copy
import numpy as np
import time
import datetime

print('Running Query...')

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
A = np.empty((rows,46))
A.fill(np.nan)

print('Building Matrix ...')

for rowIndex, row in enumerate(dataSet):
    if True:
        if ((rowIndex % 100000) == 0):
            print('Item ' + str(rowIndex) + '/' + str(rows))
        for i in range(len(row)):
            # lang case
            if i == 32:
                #print(row[i])
                if row[i] == 'java':
                    A[rowIndex][32] = 1
                    A[rowIndex][33] = 0
                    A[rowIndex][34] = 0
                if row[i] == 'javascript':
                    A[rowIndex][32] = 0
                    A[rowIndex][33] = 1
                    A[rowIndex][34] = 0
                if row[i] == 'ruby':
                    A[rowIndex][32] = 0
                    A[rowIndex][33] = 0
                    A[rowIndex][34] = 1
            elif i == 33:
                if row[i] == 'build_found':
                    A[rowIndex][35] = 1
                    A[rowIndex][36] = 0
                    A[rowIndex][37] = 0
                if row[i] == 'merge_found':
                    A[rowIndex][35] = 0
                    A[rowIndex][36] = 1
                    A[rowIndex][37] = 0
                if row[i] == 'no_previous_build':
                    A[rowIndex][35] = 0
                    A[rowIndex][36] = 0
                    A[rowIndex][37] = 1
            #dates
            elif i == 34 or i == 35:
                if row[i] != None:
                    offset = (i - 34) * 4
                    A[rowIndex][38 + offset] = row[i].timetuple()[1] #Month
                    A[rowIndex][39 + offset] = row[i].timetuple()[2] #Day
                    A[rowIndex][40 + offset] = row[i].timetuple()[3] * 3600 + row[i].timetuple()[4] * 60 + row[i].timetuple()[5] # Daytime in sec
                    A[rowIndex][41 + offset] = row[i].timetuple()[6] #Weekday
            # tr status
            elif i == 36:
                if row[i] == 'passed':
                    y[rowIndex] = 1
                else:
                    y[rowIndex] = 0
                   #A[rowIndex][i+4] = time.mktime(row[i].timetuple())
            else:
                if row[i] != None and row[i] != '':
                    A[rowIndex][i] = float(row[i])
                    
print('Finished')

np.save('A.npy', A)
np.save('y.npy', y)
