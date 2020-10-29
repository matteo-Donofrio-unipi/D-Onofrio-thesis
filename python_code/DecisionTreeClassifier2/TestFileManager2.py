import csv
from matplotlib.pyplot import errorbar
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import statistics
import os

def readCsv(fileName):

    fields = []
    rows = []


    with open(fileName, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

            # get total number of rows
        print("Total no. of rows: %d" % (csvreader.line_num))

    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields))

    #  printing first 5 rows
    print('\nFirst 5 rows are:\n')
    for row in rows:
        print(row)
        # parsing each column of a row
        # for col in row:
        #     print("%10s" % col)


def WriteCsv(fileName,row):
    fields = ['Candidates', 'Max depth', 'Min samples', 'Window size', 'Remove candi', 'k', 'useValidationSet' ,'% Training set', 'NumCluster(Medoids)' ,'Accuracy','Time']
    writeFileds=False
    if(os.path.isfile(fileName)==False):
        writeFileds=True

    # writing to csv file
    with open(fileName, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        if (writeFileds):
            csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerow(row)


def PlotValues(fileName):
    errorBarPlot=True
    # initializing the titles and rows list
    fields = []
    rows = []
    Candidates = list()
    Maxdepth = list()
    MinSamples = list()
    WindowSize = list()
    RemoveCanddates = list()
    useValidationSet=list()
    k = list()
    PercentageTrainingSet = list()
    NumClusterMedoid=list()
    Time = list()
    Accuracy = list()

    with open(fileName, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        Percentage=list()
        Accuracy=list()
        i=0
        # extracting each data row one by one
        for row in csvreader:
            if(i==0):
                i = 1
                continue
            Candidates.append(row[0])
            Maxdepth.append(row[1])
            MinSamples.append(row[2])
            WindowSize.append(row[3])
            RemoveCanddates.append(row[4])
            useValidationSet.append(row[5])
            k.append(row[6])
            PercentageTrainingSet.append(row[7])
            NumClusterMedoid.append(row[8])
            Accuracy.append(row[9])
            if(row[10]!=None):
                Time.append(row[10])
            else:
                Time.append(0)
        # get total number of rows

        dfResults = pd.DataFrame(
        columns=['Candidates', 'MaxDepth', 'MinSamples', 'WindowSize', 'RemoveCanddates', 'k', 'useValidationSet'
        'PercentageTrainingSet', 'NumClusterMedoid', 'Accuracy', 'Time'], index=range(csvreader.line_num-1))

        # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

    print(len(Candidates))
    print(csvreader.line_num)

    dfResults['Candidates']=Candidates
    dfResults['MaxDepth'] = Maxdepth
    dfResults['MinSamples'] = MinSamples
    dfResults['WindowSize'] = WindowSize
    dfResults['RemoveCanddates'] = RemoveCanddates
    dfResults['k'] = k
    dfResults['useValidationSet']=useValidationSet
    dfResults['PercentageTrainingSet'] = PercentageTrainingSet
    dfResults['NumClusterMedoid']=NumClusterMedoid
    dfResults['Accuracy'] = Accuracy
    dfResults['Time'] = Time



    dfResults['Accuracy'] = list(map(float, dfResults['Accuracy']))

    percentage=range(10,110,10)
    mean=list()
    stdevList=list()

    dfResults = dfResults.sort_values(by='Candidates', ascending=True)
    print(dfResults['Candidates'])

    plt.plot(dfResults['Candidates'], dfResults['Accuracy'], 'or')
    plt.xlabel('Candidates')
    plt.ylabel('Accuracy')
    plt.show()


    dfResults = dfResults.sort_values(by='MaxDepth', ascending=True)
    print(dfResults['WindowSize'])

    plt.plot(dfResults['MaxDepth'], dfResults['Accuracy'], 'or')
    plt.xlabel('MaxDepth')
    plt.ylabel('Accuracy')
    plt.show()

    dfResults = dfResults.sort_values(by='WindowSize', ascending=True)
    print(dfResults)

    plt.plot(dfResults['WindowSize'], dfResults['Accuracy'], 'or')
    plt.xlabel('WindowSize')
    plt.ylabel('Accuracy')
    plt.show()

    dfResults = dfResults.sort_values(by='NumClusterMedoid', ascending=True)
    print(dfResults)

    plt.plot(dfResults['NumClusterMedoid'], dfResults['Accuracy'], 'or')
    plt.xlabel('NumClusterMedoid ')
    plt.ylabel('Accuracy')
    plt.show()

    if(errorBarPlot):


        actualP=dfResults.iloc[0]['PercentageTrainingSet']
        actualMean=list()
        actualMean.append(dfResults.iloc[0]['Accuracy'])

        for i in range (len(dfResults)):
            if(i==0):
                continue
            if(dfResults.iloc[i]['PercentageTrainingSet']==actualP and i!=(len(dfResults)-1)):
                actualMean.append(dfResults.iloc[i]['Accuracy'])
                actualP=dfResults.iloc[i]['PercentageTrainingSet']
                continue
            else:
                mean.append(sum(actualMean)/len(actualMean))
                if(len(actualMean)>1):
                    stdevList.append(statistics.stdev(actualMean))
                else:
                    stdevList.append(0.1)
                actualMean.clear()
                actualMean.append(dfResults.iloc[i]['Accuracy'])
                actualP = dfResults.iloc[i]['PercentageTrainingSet']

        print(mean)
        print(stdevList)



        plt.errorbar(percentage, mean, yerr=stdevList, fmt='.k')
        plt.xlabel('% Training Set')
        plt.ylabel('Mean')
        plt.show()

