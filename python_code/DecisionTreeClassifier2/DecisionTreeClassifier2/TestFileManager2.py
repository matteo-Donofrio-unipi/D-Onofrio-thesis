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


def readCsvAsDf(fileName):
    Candidates = list()
    Maxdepth = list()
    MinSamples = list()
    WindowSize = list()
    RemoveCanddates = list()
    useValidationSet = list()
    k = list()
    PercentageTrainingSet = list()
    useClustering = list()
    NumClusterMedoid = list()
    Time = list()

    with open(fileName, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        Percentage = list()
        Accuracy = list()
        i = 0
        # extracting each data row one by one
        for row in csvreader:
            if (i == 0):
                i = 1
                continue
            Candidates.append(row[0])
            Maxdepth.append(row[1])
            MinSamples.append(row[2])
            WindowSize.append(row[3])
            RemoveCanddates.append(row[4])
            k.append(row[5])
            useValidationSet.append(row[6])
            PercentageTrainingSet.append(row[7])
            useClustering.append(row[8])
            NumClusterMedoid.append(row[9])
            Accuracy.append(row[10])
            if (row[11] != None):
                Time.append(row[11])
            else:
                Time.append(0)
        # get total number of rows

        dfResults = pd.DataFrame(
            columns=['Candidates', 'MaxDepth', 'MinSamples', 'WindowSize', 'RemoveCanddates', 'k', 'useValidationSet'
                                                                                                   'PercentageTrainingSet',
                     'useClustering', 'NumClusterMedoid', 'Accuracy', 'Time'], index=range(csvreader.line_num - 1))


    dfResults['Candidates'] = Candidates
    dfResults['MaxDepth'] = Maxdepth
    dfResults['MinSamples'] = MinSamples
    dfResults['WindowSize'] = WindowSize
    dfResults['RemoveCandidates'] = RemoveCanddates
    dfResults['k'] = k
    dfResults['useValidationSet'] = useValidationSet
    dfResults['PercentageTrainingSet'] = PercentageTrainingSet
    dfResults['useClustering'] = useClustering
    dfResults['NumClusterMedoid'] = NumClusterMedoid
    dfResults['Accuracy'] = Accuracy
    dfResults['Time'] = Time

    dfResults['Accuracy'] = list(map(float, dfResults['Accuracy']))

    return dfResults


def WriteCsv(fileName,row):
    fields = ['Candidates', 'Max depth', 'Min samples', 'Window size', 'Remove candi', 'k', 'useValidationSet' ,'% Training set', 'useClustering','NumCluster(Medoids)' ,'Accuracy','Time']
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



def WriteCsvComparison(fileName,row):
    fields = ['Algorithm', 'DatasetName', 'Accuracy', 'Time']
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

#stampa parametri VS accuracy fissato un dataset
def PlotValues(fileName):
    errorBarPlot=True
    dfResults=readCsvAsDf(fileName)

    percentage=range(10,110,10)
    mean=list()
    stdevList=list()

    dfResults = dfResults.sort_values(by='Candidates', ascending=True)
    print(dfResults['Candidates'])

    # plt.plot(dfResults['Candidates'], dfResults['Accuracy'], 'or')
    # plt.xlabel('Candidates')
    # plt.ylabel('Accuracy')
    # plt.show()
    #
    #
    # dfResults = dfResults.sort_values(by='MaxDepth', ascending=True)
    # print(dfResults['WindowSize'])
    #
    # plt.plot(dfResults['MaxDepth'], dfResults['Accuracy'], 'or')
    # plt.xlabel('MaxDepth')
    # plt.ylabel('Accuracy')
    # plt.show()
    #
    # dfResults = dfResults.sort_values(by='WindowSize', ascending=True)
    # print(dfResults)
    #
    # plt.plot(dfResults['WindowSize'], dfResults['Accuracy'], 'or')
    # plt.xlabel('WindowSize')
    # plt.ylabel('Accuracy')
    # plt.show()
    #
    # dfResults = dfResults.sort_values(by='NumClusterMedoid', ascending=True)
    # print(dfResults)
    #
    # plt.plot(dfResults['NumClusterMedoid'], dfResults['Accuracy'], 'or')
    # plt.xlabel('NumClusterMedoid ')
    # plt.ylabel('Accuracy')
    # plt.show()

    if(errorBarPlot):

        dfResults = dfResults.sort_values(by='PercentageTrainingSet', ascending=True)

        actualP=dfResults.iloc[0]['PercentageTrainingSet']
        actualMean=list()
        actualMean.append(dfResults.iloc[0]['Accuracy'])

        for i in range (len(dfResults)):
            if(i==0 or dfResults.iloc[i]['useValidationSet']==True):
                continue
            if(dfResults.iloc[i]['PercentageTrainingSet']==actualP):
                actualMean.append(dfResults.iloc[i]['Accuracy'])
                actualP=dfResults.iloc[i]['PercentageTrainingSet']
                continue
            else:
                mean.append(sum(actualMean)/len(actualMean))
                if(len(actualMean)>1):
                    stdevList.append(statistics.stdev(actualMean))
                else:
                    stdevList.append(0) #default case
                actualMean.clear()
                actualMean.append(dfResults.iloc[i]['Accuracy'])
                actualP = dfResults.iloc[i]['PercentageTrainingSet']

        #computo ultinmo step
        mean.append(sum(actualMean) / len(actualMean))
        if (len(actualMean) > 1):
            stdevList.append(statistics.stdev(actualMean))
        else:
            stdevList.append(0)  # default case


        print(mean)
        print(stdevList)

        percentage=dfResults['PercentageTrainingSet'].values
        percentage=np.unique(percentage)
        print(percentage)

        plt.errorbar(percentage, mean, yerr=stdevList, fmt='.k')
        plt.xlabel('% Training Set')
        plt.ylabel('Mean')
        plt.show()



#plot del dataset al variare di due attributi
def plotComparisonMultiple(fileName,datasetName,attribute1,attribute2,mOa):
    #mOa=0 => max ||  mOa=1 => avg
    #ATT1 SU ASSE X, ATT2 SU CUI EFFETTO COMPARAZIONE -> PRENDO DIVERSE ACCURACY AL VARIARE DEL VALORE DI TALE ATTRIBUTO
    dfResults=readCsvAsDf(fileName)

    print(dfResults)

    dftest=dfResults[dfResults["useValidationSet"]=="False"]

    #prendo differenti valori dell'attributo su asse x
    valuesAtt1=np.unique(dftest[attribute1].values)





    # prendo differenti valori dell'attributo da confrontare
    valuesAtt2 = np.unique(dftest[attribute2].values)


    print('ATT')
    print(valuesAtt1)
    print(valuesAtt2)

    colori = 0
    colors = 'rbgcmyk'

    accuracyList=[]
    plt.title(datasetName)

    for i in range(len(valuesAtt2)):
        valueAtt2=valuesAtt2[i]

        c = colors[colori % len(colors)]

        #fissato valore att2, scandisco tutti i valori di att1 (asse x) e faccio media/best accuracy
        for j in range(len(valuesAtt1)):

            valueAtt1=valuesAtt1[j]

            #acc migliore per att2 fissato e variazione di att1
            #dfLocal=dftest[(dftest[attribute2]==valueAtt2) & (dftest[attribute1]==valueAtt1)]
            dfLocal=dftest[(dftest[attribute2]==valueAtt2)]
            dfLocal=dfLocal[(dfLocal[attribute1])==valueAtt1]['Accuracy']
            if(len(dfLocal)>0):
                accuracy=dfLocal.values
                if(mOa==0):
                    choosenAccuracy=max(accuracy)
                else:
                    choosenAccuracy = sum(accuracy)/len(accuracy)
            else:
                choosenAccuracy=0


            accuracyList.append(choosenAccuracy)

        #ora accuracyList ha tanti valori quanti i possibili valori di att1, fissato il valore di att2

        plt.plot(valuesAtt1, accuracyList, color=c, marker='o', label=attribute2+'= '+str(valueAtt2))
        accuracyList.clear()
        colori += 1

    plt.xlabel(attribute1)
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(datasetName + '-' + attribute1 + '-' + attribute2 + '-' + '.pdf')
    plt.show()

#plot del dataset, fissati n-1 attrivuti, ne vario 1
def plotComparisonSingle(fileName,datasetName,attribute1,mOa,UsePercentageTrainingSet):
    #mOa=0 => max ||  mOa=1 => avg
    #ATT1 SU ASSE X, ATT2 SU CUI EFFETTO COMPARAZIONE -> PRENDO DIVERSE ACCURACY AL VARIARE DEL VALORE DI TALE ATTRIBUTO
    dfResults=readCsvAsDf(fileName)

    print(dfResults)

    dftest=dfResults[dfResults["useValidationSet"]=="False"]

    dfLocal=dftest
    # SCELGO LA CONFIGURAZIONE MIGLIORE
    dfLocal = dfLocal[(dftest['Candidates'] == 'Discords')]
    dfLocal = dfLocal[(dfLocal['MaxDepth'] == '3')]
    dfLocal = dfLocal[(dfLocal['MinSamples'] == '20')]
    dfLocal = dfLocal[(dfLocal['WindowSize'] == '20')]
    dfLocal = dfLocal[(dfLocal['RemoveCandidates'] == '1')]
    dfLocal = dfLocal[(dfLocal['k'] == '2')]
    #dfLocal = dfLocal[(dfLocal['NumClusterMedoid'] == '20')]

    print(dfLocal)

    if (UsePercentageTrainingSet): #su asse x metto Percentage
        attribute1='PercentageTrainingSet'
    else:
        dfLocal = dfLocal[dfLocal['PercentageTrainingSet'] == '1'] #prendo training set interi


    #prendo differenti valori dell'attributo su asse x
    valuesAtt1=np.unique(dfLocal[attribute1].values)
    valuesAtt1=valuesAtt1.astype(float)
    valuesAtt1=sorted(valuesAtt1)
    for i in range(len(valuesAtt1)):
        valuesAtt1[i]=str(valuesAtt1[i])

    if(UsePercentageTrainingSet): #setto 1 cosi riesco a fare la query correttamente, sul file è memorizzato come 1
        valuesAtt1[-1]='1'


    print('ATT')
    print(valuesAtt1)
    colori = 0
    colors = 'rbgcmyk'

    accuracyList=[]
    timeList=[]
    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.set_title(datasetName)

    for i in range(len(valuesAtt1)):
        valueAtt1=valuesAtt1[i]

        c = colors[colori % len(colors)]

        accuracy=dfLocal[(dfLocal[attribute1])==valueAtt1]['Accuracy'].values
        time=dfLocal[(dfLocal[attribute1])==valueAtt1]['Time'].values
        time=time.astype(float)

        if (len(accuracy) > 0):
            if (mOa == 0):
                choosenAccuracy = max(accuracy)
                choosenTime=max(time)
            else:
                choosenTime=sum(time)/len(time)
                choosenAccuracy = sum(accuracy) / len(accuracy)
        else:
            choosenAccuracy = 0

        timeList.append(choosenTime)
        accuracyList.append(choosenAccuracy)

        # ora accuracyList ha tanti valori quanti i possibili valori di att1, fissato il valore di att2


    ax1.plot(valuesAtt1, accuracyList, color='r', marker='o')
    ax2.plot(valuesAtt1, timeList, color='b', marker='^')


    ax1.set_xlabel(attribute1)
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Time')


    plt.savefig(datasetName + '-' + attribute1 + '-' + '.pdf')
    plt.show()






def buildTable(fileName,datasetName,query):

    dfResult=readCsvAsDf(fileName)

    print(dfResult)

    dfResult=dfResult[(dfResult['useValidationSet']=="False")]

    print(dfResult)

    #['Motifs','3','10','5','1','2','40']
    dfLocal = dfResult[(dfResult['Candidates'] == query[0])]
    dfLocal = dfLocal[(dfLocal['MaxDepth'] == query[1])]
    print('prinop')
    print(dfLocal)
    dfLocal = dfLocal[(dfLocal['MinSamples'] == query[2])]
    dfLocal = dfLocal[(dfLocal['WindowSize'] == query[3])]
    dfLocal = dfLocal[(dfLocal['RemoveCandidates'] == query[4])]
    dfLocal = dfLocal[(dfLocal['k'] == query[5])]
    dfLocal = dfLocal[(dfLocal['NumClusterMedoid'] == query[6])]

    print(dfLocal)

    accuracy = dfLocal['Accuracy'].values
    time = dfLocal['Time'].values


    accuracy=max(accuracy)
    time=min(time)

    row=[datasetName,accuracy,time]

    fields = ['NameDataset', 'Accuracy', 'Execution Time']
    writeFileds = False
    if (os.path.isfile(fileName) == False):
        writeFileds = True

    # writing to csv file
    with open(fileName, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        if (writeFileds):
            csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerow(row)











