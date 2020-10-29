from Tree2 import *
from PreProcessingLibrary2 import  *
from TestFileManager2 import *
import time
from sklearn.utils.random import sample_without_replacement
from tslearn.datasets import UCR_UEA_datasets
from pathlib import Path
from datetime import datetime

#datasetNames = 'GunPoint,ItalyPowerDemand,ArrowHead,ECG200,ECG5000,ElectricDevices,PhalangesOutlinesCorrect'
def executeTest(useValidationSet,usePercentageTrainingSet,datasetName,nameFile):

    first = True  # ESTRAZIONE DATASET TRAINING
    second = True  # CALCOLO ALBERO DECISIONE
    third = True  # ESTRAZIONE DATASET TEST
    quarter = True  # PREDIZIONE E RISULTATO
    fifth = False  # GRAFICA DI SERIE TEMPORALI E MATRIX PROFILE DEI CANDIDATI SCELTI

#METTERE K HYPER PARAMETRO CENTROIDI COME PARAMETRO DI TREE


    PercentageTrainingSet = 0.3   # % se voglio usare una percentuale di Training Set
    PercentageValidationSet = 0.3  # % set rispetto alla dim del Training Set
    writeOnCsv = True


    #genero albero (VUOTO) e avvio timer
    le = LabelEncoder()
    tree= Tree(candidatesGroup=0,maxDepth=3,minSamplesLeaf=10,removeUsedCandidate=0,window_size=15,k=2,useClustering=True,n_clusters=20,warningDetected=False,verbose=1) # K= NUM DI MOTIF/DISCORD ESTRATTI

    start_time = time.time()




    if(first==True):
        verbose = True


        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)
        dimWholeTrainSet = len(X_train)
        print('Initial Train set shape : ' + str(X_train.shape)+'\n')
        print('Initial Test set shape : ' + str(X_test.shape) + '\n')

        if(useValidationSet):



            dimValidationSet = int(len(X_train) * PercentageValidationSet)  # dim of new SubSet of X_train
            selectedRecordsForValidation=sample_without_replacement(len(X_train), dimValidationSet)
            print('selectedRecordsForValidation: '+str(selectedRecordsForValidation)+'\n')

            # inserisco in df Training set con relative label
            dfTrain = computeLoadedDataset(X_train, y_train)

            patternLenght=len(dfTrain.iloc[0])-1


            #estraggo val set e rimuovo record estratti da training set
            dfVal=dfTrain.iloc[selectedRecordsForValidation]
            dfTrain=dfTrain.drop(index=selectedRecordsForValidation)


            print('Patter Lenght: ' + str(patternLenght) + '\n')

            print('Final Train set shape : ' + str(dfTrain.shape))
            print('Final Validation set shape : '+ str(dfVal.shape)+'\n')

            num_classes = le.fit_transform(dfVal['target'])
            print('Final class distribution in Validation set : ')
            print(np.unique(num_classes, return_counts=True))
            print('\n')

            num_classes = le.fit_transform(dfTrain['target'])
            print('Final class distribution in Training set : ')
            print(np.unique(num_classes, return_counts=True))
            print('\n')

            print('dfTrain: \n'+str(dfTrain))
            print(dfTrain.isnull().sum().sum())
            print(dfTrain.isnull().values.any())
            print('dfVal: \n'+str(dfVal))


        if(usePercentageTrainingSet):

            dimSubTrainSet = int(len(X_train) * PercentageTrainingSet)  # dim of new SubSet of X_train
            selectedRecords = sample_without_replacement(len(X_train), dimSubTrainSet)  # random records selected
            print('selectedRecords: '+str(selectedRecords))

            #inserisco in df Training set con relative label
            dfTrain = computeLoadedDataset(X_train, y_train)

            patternLenght = len(dfTrain.iloc[0]) - 1


            dfTrain = dfTrain.iloc[selectedRecords].copy()

            print('Final Train set shape : ' + str(dfTrain.shape)+'\n')

            num_classes = le.fit_transform(dfTrain['target'])
            print('Final class distribution in Training set : ')
            print(np.unique(num_classes, return_counts=True))
            print('\n')

            print('PATT LENGHT: ' + str(patternLenght))

        # genero strtutture dati utilizzate effettivamente
        tree.dfTrain = dfTrain
        mpTrain, OriginalCandidatesListTrain, numberOfMotifTrain, numberOfDiscordTrain  = getDataStructures(tree,
            dfTrain, tree.window_size, tree.k, verbose=1)


        #sfoltisco i candidati in base al gruppo di candidati scelti
        if(tree.candidatesGroup==0):
            OriginalCandidatesListTrain=OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D']==0]
        if (tree.candidatesGroup == 1):
            OriginalCandidatesListTrain = OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D'] == 1]

        OriginalCandidatesListTrain.reset_index(drop=True)

        #aggiungo lista candidati e lista candidati usati, ORIGINALI, in tree
        tree.OriginalCandidatesUsedListTrain = buildCandidatesUsedList(OriginalCandidatesListTrain)
        tree.OriginalCandidatesListTrain=OriginalCandidatesListTrain
        print('OriginalCandidatesUsedListTrain: \n')
        print(tree.OriginalCandidatesUsedListTrain)

        print('OriginalCandidatesListTrain: \n')
        print(tree.OriginalCandidatesListTrain)


        #OriginalCandidatesListTrain VIENE MANTENUTO PER TUTTO L'ALGORITMO, DA ESSO AD OGNI SPLIT PRENDO CANDIDATI NECESSARI



        #PREPARO STRUTTURE DATI PER LA PRIMA ITERAZIONE DI FIT
        #applico clustering a insieme di candidati iniziali
        CandidatesListTrain = reduceNumberCandidates(tree, OriginalCandidatesListTrain,returnOnlyIndex=False)
        print('candidati rimasti/ più significativi-distintivi ')
        print(CandidatesListTrain)


        TsIndexList=dfTrain['TsIndex'].values #inizialmente tutto DfTrain ( prima iterazione )

        # computeSubSeqDistance calcola distanze tra lista di Ts e lista di candidati fornite
        dfForDTree = computeSubSeqDistance(tree,TsIndexList, CandidatesListTrain, tree.window_size)
        if (verbose == True):
            print('dfTrain: \n'+str(dfTrain))
            print('dfForDTree: \n'+str(dfForDTree))

        print("--- %s seconds after getting DATA STRUCTURES" % (time.time() - start_time))


    if(second==True):
        verbose = True
        #COSTRUISCO DECISION TREE
        tree.fit(dfForDTree,verbose=True)
        if(verbose==True):
            print(tree.attributeList)
            print(tree.Root)
            tree.printAll(tree.Root)

        print("--- %s seconds after building TREE" % (time.time() - start_time))



    if(third==True):
        verbose=True

        if(useValidationSet):
            dfTest=dfVal
        else:
            dfTest = computeLoadedDataset(X_test, y_test)

        print('DF TEST')
        print(dfTest)

        tree.attributeList=sorted(tree.attributeList) #ordino attributi per rendere più efficiente 'computeSubSeqDistanceForTest'
        tree.attributeList=np.unique(tree.attributeList)

        CandidatesListMatched = tree.OriginalCandidatesListTrain['IdCandidate'].isin(
            tree.attributeList)  # mi dice quali TsIndex in OriginalCandidatesListTrain sono contenuti in Dleft

        tree.dTreeAttributes = tree.OriginalCandidatesListTrain[
            CandidatesListMatched]  # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dleft

        print('Attributi selezionati dal Decision Tree')
        print(tree.dTreeAttributes)

        dfForDTreeTest=computeSubSeqDistanceForTest(tree,dfTest,tree.dTreeAttributes)
        if(verbose==True):
            print(dfForDTreeTest)




    if(quarter==True):
        # EFFETTUO PREDIZIONE E MISURO RISULTATO
        verbose = True
        yTest, yPredicted = tree.predict(dfForDTreeTest, tree.Root)

        if (verbose == True):
            for a, b in zip(yTest, yPredicted):
                print(a, b)

        cR = classification_report(yTest, yPredicted)
        aS = accuracy_score(yTest, yPredicted)
        f1 = f1_score(yTest, yPredicted, average=None)
        confusion_matrix(yTest, yPredicted)
        totalTime = time.time() - start_time

        if (tree.candidatesGroup == 0):
            group = 'Motifs'
        elif (tree.candidatesGroup == 1):
            group = 'Discords'
        else:
            group = 'Both'

        if(useValidationSet):
            percentage=PercentageValidationSet
        elif(usePercentageTrainingSet):
            percentage=PercentageTrainingSet

        row=[group,tree.maxDepth,tree.minSamplesLeaf,tree.window_size,tree.removeUsedCandidate,tree.k,useValidationSet,percentage,tree.n_clusters,round(aS,2),round(totalTime,2)]
        print('Classification Report %s' % cR)
        print('Accuracy %s' % aS)
        print('F1-score %s' % f1)
        print(" %s seconds END OF EXECUTION" % totalTime)

        if(writeOnCsv):
            WriteCsv(nameFile, row)


    if(fifth==True):

        #ESTRAGGO TUTTO DI NUOVO PERCHE LE TS USATE PRIMA HANNO VALORI AGGIUNTI
        for i in range(len(tree.dTreeAttributes)):
            idTs=tree.dTreeAttributes.iloc[i]['IdTs']
            idCandidate=tree.dTreeAttributes.iloc[i]['IdCandidate']
            sp = tree.dTreeAttributes.iloc[i]['startingPosition']
            md=tree.dTreeAttributes.iloc[i]['M/D']
            ts = np.array(tree.dfTrain[tree.dfTrain['TsIndex'] == idTs].values)
            ts=ts[0]
            ts = ts[:len(ts) - 2]


            tupla=retrieve_all(tree,ts,tree.window_size,tree.k)

            mp, mot, motif_dist, dis =tupla

            print('IdTs:  %d' % idTs)
            print('IDCandidate:  %d' % idCandidate)
            print('starting position:  %d ' % sp)
            print('M/D: %d ' % md)

            plot_all(ts, mp, mot, motif_dist, dis, tree.window_size)



def plotTs(datasetName):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    dfTrain = computeLoadedDataset(X_train, y_train)

    le = LabelEncoder()
    num_classes = le.fit_transform(dfTrain['target'])
    plt.scatter(dfTrain['att0'], dfTrain['att1'],
                c=num_classes)  # scatter mi permette di "disegnare" il piano in 2d, mettendo attributi, e avere graficamente classificazione lineare
    plt.show()

    for i in range(len(dfTrain)):
        Ts = np.array(dfTrain.iloc[i].values)
        print('TS ID:' + str(i))
        print('TS CLASS:' + str(dfTrain.iloc[i]['target']))
        plotData(dfTrain.iloc[i])


def plotTestResults(nameFile):

    PlotValues(nameFile)



