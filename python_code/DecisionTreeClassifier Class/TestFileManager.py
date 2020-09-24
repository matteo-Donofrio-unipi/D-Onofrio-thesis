import csv
from matplotlib.pyplot import errorbar
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import statistics

def readCsv(fileName):
    # csv file name

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
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
    #fields = ['Candidates', 'Max depth', 'Min samples', 'Window size', 'Remove candi', 'k', '% Training set', 'Accuracy']

    # writing to csv file
    with open(fileName, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the data rows
        csvwriter.writerow(row)


def PlotValues(fileName):
    # initializing the titles and rows list
    fields = []
    rows = []
    Candidates = list()
    Maxdepth = list()
    MinSamples = list()
    WindowSize = list()
    RemoveCanddates = list()
    k = list()
    PercentageTrainingSet = list()
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
            k.append(row[5])
            PercentageTrainingSet.append(row[6])
            Accuracy.append(row[7])
            if(row[8]!=None):
                Time.append(row[8])
            else:
                Time.append(0)
        # get total number of rows

        dfResults = pd.DataFrame(
        columns=['Candidates', 'MaxDepth', 'MinSamples', 'WindowSize', 'RemoveCanddates', 'k',
        'PercentageTrainingSet', 'Accuracy', 'Time'], index=range(csvreader.line_num-1))

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
    dfResults['PercentageTrainingSet'] = PercentageTrainingSet
    dfResults['Accuracy'] = Accuracy
    dfResults['Time'] = Time



    dfResults['Accuracy'] = list(map(float, dfResults['Accuracy']))

    percentage=range(10,110,10)
    mean=list()
    stdevList=list()

    dfResults = dfResults.sort_values(by='PercentageTrainingSet', ascending=True)
    print(dfResults)


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

    plt.plot(dfResults['PercentageTrainingSet'], dfResults['Accuracy'], 'or')
    plt.xlabel('% Training Set')
    plt.ylabel('Accuracy')
    plt.show()

    plt.errorbar(percentage, mean, yerr=stdevList, fmt='.k')
    plt.xlabel('% Training Set')
    plt.ylabel('Mean')
    plt.show()

    dfResults = dfResults.sort_values(by='MaxDepth', ascending=True)
    print(dfResults['WindowSize'])

    plt.plot(dfResults['MaxDepth'], dfResults['Accuracy'], 'or')
    plt.xlabel('MaxDepth')
    plt.ylabel('Accuracy')
    plt.show()