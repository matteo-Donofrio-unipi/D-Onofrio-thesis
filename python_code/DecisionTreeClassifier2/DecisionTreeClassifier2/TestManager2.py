from Tree2 import *
from PreProcessingLibrary2 import  *
from TestFileManager2 import *
import time
from sklearn.utils.random import sample_without_replacement
from tslearn.datasets import UCR_UEA_datasets
from sklearn import tree as SkTree
from pathlib import Path
from datetime import datetime
from pyts.transformation import ShapeletTransform


#datasetNames = 'GunPoint,ItalyPowerDemand,ArrowHead,ECG200,ECG5000,ElectricDevices,PhalangesOutlinesCorrect'
def executeTest(useValidationSet,usePercentageTrainingSet,datasetName,nameFile):

    first = True  # ESTRAZIONE DATASET TRAINING
    second = True  # CALCOLO ALBERO DECISIONE
    third = True  # ESTRAZIONE DATASET TEST
    quarter = True  # PREDIZIONE E RISULTATO
    fifth = False  # GRAFICA DI SERIE TEMPORALI E MATRIX PROFILE DEI CANDIDATI SCELTI

#METTERE K HYPER PARAMETRO CENTROIDI COME PARAMETRO DI TREE


    PercentageTrainingSet = 0.9 # % se voglio usare una percentuale di Training Set
    PercentageValidationSet = 0.3  # % set rispetto alla dim del Training Set
    writeOnCsv = True


    #genero albero (VUOTO) e avvio timer
    le = LabelEncoder()
    tree= Tree(candidatesGroup=1,maxDepth=3,minSamplesLeaf=20,removeUsedCandidate=1,window_size=20,k=2,useClustering=True,n_clusters=20,warningDetected=False,verbose=0) # K= NUM DI MOTIF/DISCORD ESTRATTI





    if(first==True):
        verbose = False


        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)


        if(verbose):
            print('Initial Train set shape : ' + str(X_train.shape)+'\n')
            print('Initial Test set shape : ' + str(X_test.shape) + '\n')

        if(useValidationSet):



            dimValidationSet = int(len(X_train) * PercentageValidationSet)  # dim of new SubSet of X_train
            selectedRecordsForValidation=sample_without_replacement(len(X_train), dimValidationSet)
            #print('selectedRecordsForValidation: '+str(selectedRecordsForValidation)+'\n')

            # inserisco in df Training set con relative label
            dfTrain = computeLoadedDataset(X_train, y_train)

            patternLenght=len(dfTrain.iloc[0])-1


            #estraggo val set e rimuovo record estratti da training set
            dfVal=dfTrain.iloc[selectedRecordsForValidation]
            dfTrain=dfTrain.drop(index=selectedRecordsForValidation)


            if(verbose):

                print('Patter Lenght: ' + str(patternLenght) + '\n')
                print('Final Train set shape : ' + str(dfTrain.shape))
                print('Final Validation set shape : '+ str(dfVal.shape)+'\n')

            num_classes = le.fit_transform(dfVal['target'])

            if (verbose):
                print('Final class distribution in Validation set : ')
                print(np.unique(num_classes, return_counts=True))
                print('\n')

            num_classes = le.fit_transform(dfTrain['target'])

            if (verbose):
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

            if (verbose):
                print('selectedRecords: '+str(selectedRecords))

            #inserisco in df Training set con relative label
            dfTrain = computeLoadedDataset(X_train, y_train)

            patternLenght = len(dfTrain.iloc[0]) - 1


            dfTrain = dfTrain.iloc[selectedRecords].copy()

            if (verbose):
                print('Final Train set shape : ' + str(dfTrain.shape)+'\n')

            num_classes = le.fit_transform(dfTrain['target'])

            if (verbose):
                print('Final class distribution in Training set : ')
                print(np.unique(num_classes, return_counts=True))
                print('\n')

                print('PATT LENGHT: ' + str(patternLenght))

        # genero strtutture dati utilizzate effettivamente
        start_time = time.time()
        tree.dfTrain = dfTrain
        mpTrain, OriginalCandidatesListTrain, numberOfMotifTrain, numberOfDiscordTrain  = getDataStructures(tree,
            dfTrain, tree.window_size, tree.k, verbose)


        #sfoltisco i candidati in base al gruppo di candidati scelti
        if(tree.candidatesGroup==0):
            OriginalCandidatesListTrain=OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D']==0]
        if (tree.candidatesGroup == 1):
            OriginalCandidatesListTrain = OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D'] == 1]

        OriginalCandidatesListTrain.reset_index(drop=True)

        #aggiungo lista candidati e lista candidati usati, ORIGINALI, in tree
        tree.OriginalCandidatesUsedListTrain = buildCandidatesUsedList(OriginalCandidatesListTrain)
        tree.OriginalCandidatesListTrain=OriginalCandidatesListTrain
        if (verbose):
            print('OriginalCandidatesUsedListTrain: \n')
            print(tree.OriginalCandidatesUsedListTrain)

            print('OriginalCandidatesListTrain: \n')
            print(tree.OriginalCandidatesListTrain)


        #OriginalCandidatesListTrain VIENE MANTENUTO PER TUTTO L'ALGORITMO, DA ESSO AD OGNI SPLIT PRENDO CANDIDATI NECESSARI



        #PREPARO STRUTTURE DATI PER LA PRIMA ITERAZIONE DI FIT
        #applico clustering a insieme di candidati iniziali
        if(tree.useClustering):
            CandidatesListTrain = reduceNumberCandidates(tree, OriginalCandidatesListTrain,returnOnlyIndex=False)
            if (verbose):
                print('candidati rimasti/ più significativi-distintivi ')
        else:
            CandidatesListTrain=tree.OriginalCandidatesListTrain
        if (verbose):
            print(CandidatesListTrain)


        TsIndexList=dfTrain['TsIndex'].values #inizialmente tutto DfTrain ( prima iterazione )

        # computeSubSeqDistance calcola distanze tra lista di Ts e lista di candidati fornite
        dfForDTree = computeSubSeqDistance(tree,TsIndexList, CandidatesListTrain, tree.window_size)
        if (verbose == True):
            print('dfTrain: \n'+str(dfTrain))
            print('dfForDTree: \n'+str(dfForDTree))

        #print("--- %s seconds after getting DATA STRUCTURES" % (time.time() - start_time))


    if(second==True):
        verbose = False
        #COSTRUISCO DECISION TREE
        tree.fit(dfForDTree,verbose=False)
        fitTime=time.time() - start_time
        if(verbose==True):
            print(tree.attributeList)
            print(tree.Root)
            tree.printAll(tree.Root)

        print("--- %s seconds after fitting" % (fitTime))



    if(third==True):
        verbose=True

        if(useValidationSet):
            dfTest=dfVal
        else:
            dfTest = computeLoadedDataset(X_test, y_test)

        if(tree.verbose):
            print('DF TEST')
            print(dfTest)

        tree.attributeList=sorted(tree.attributeList) #ordino attributi per rendere più efficiente 'computeSubSeqDistanceForTest'
        tree.attributeList=np.unique(tree.attributeList)

        CandidatesListMatched = tree.OriginalCandidatesListTrain['IdCandidate'].isin(
            tree.attributeList)  # mi dice quali TsIndex in OriginalCandidatesListTrain sono contenuti in Dleft

        tree.dTreeAttributes = tree.OriginalCandidatesListTrain[
            CandidatesListMatched]  # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dleft

        if(tree.verbose):
            print('Attributi selezionati dal Decision Tree')
            print(tree.dTreeAttributes)

        dfForDTreeTest=computeSubSeqDistanceForTest(tree,dfTest,tree.dTreeAttributes)
        if(tree.verbose==True):
            print(dfForDTreeTest)




    if(quarter==True):
        # EFFETTUO PREDIZIONE E MISURO RISULTATO
        verbose = True

        #INSERISCO LA SERIE PER STAMPARLA DOPO CLASSIFICAZIONE
        tree.TsTestForPrint = dfTest.iloc[0].values  # contiene la prima serie che viene classificata
        tree.TsTestForPrint = tree.TsTestForPrint[:len(tree.TsTestForPrint) - 2]

        yTest, yPredicted = tree.predict(dfForDTreeTest, tree.Root)

        totalTime = time.time() - start_time


        if (tree.verbose == True):
            for a, b in zip(yTest, yPredicted):
                print(a, b)

        cR = classification_report(yTest, yPredicted)
        aS = accuracy_score(yTest, yPredicted)
        f1 = f1_score(yTest, yPredicted, average=None)
        confusion_matrix(yTest, yPredicted)


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

        row=[group,tree.maxDepth,tree.minSamplesLeaf,tree.window_size,tree.removeUsedCandidate,tree.k,useValidationSet,percentage,tree.useClustering,tree.n_clusters,round(aS,2),round(totalTime,2)]
        print('Classification Report  \n%s ' % cR)
        print('Accuracy %s' % aS)
        print('F1-score %s' % f1)
        print(" %s seconds END OF EXECUTION" % totalTime)

        #COMMENTO PER STAMPARE SU CONFRONTO ALGO
        if(writeOnCsv):
            WriteCsv(nameFile, row)
        # row2=['MatrixProfileBased',datasetName,aS,totalTime]
        # WriteCsvComparison('confrontoAlgoritmi.csv',row2)

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

            plot_all(ts, mp, mot, motif_dist, dis, sp, md,tree.window_size,idCandidate)



def executeShapeletTransform(datasetName):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    #RE-SIZE BY FUN X TRAIN
    dfTrain = computeLoadedDataset(X_train, y_train)

    y_train=dfTrain['target'].values
    y_train=y_train.astype(int)

    del dfTrain['target']
    del dfTrain['TsIndex']

    # RE-SIZE BY FUN X TEST
    dfTest=computeLoadedDataset(X_test,y_test)

    y_test = dfTest['target'].values
    y_test = y_test.astype(int)

    del dfTest['target']
    del dfTest['TsIndex']


    start_time = time.time()

    #Shapelet transformation WITH RANDOM STATE
    st = ShapeletTransform(window_sizes=[15],
                           random_state=41, sort=True)
    X_new = st.fit_transform(dfTrain, y_train)
    X_test_new = st.transform(dfTest)

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier()
    clf.fit(X_new, y_train)

    timeToFit = time.time() - start_time

    print("--- %s seconds after fitting" % (timeToFit))

    y_pred = clf.predict(X_test_new)

    timeToTest = time.time() - start_time

    print(accuracy_score(y_pred, y_test))

    print("--- %s seconds after testing" % (timeToTest))

    row=['ShapeletTransformation',datasetName,accuracy_score(y_pred, y_test),timeToTest]

    WriteCsvComparison('confrontoAlgoritmi.csv',row)




def executeClassicDtree(datasetName):

    #INIZIO STESSA PROCEDURA EPR GENERARE dfForDTree
    tree= Tree(candidatesGroup=1,maxDepth=3,minSamplesLeaf=100,removeUsedCandidate=1,window_size=15,k=1,useClustering=True,n_clusters=20,warningDetected=False,verbose=0) # K= NUM DI MOTIF/DISCORD ESTRATTI

    verbose=False

    le = LabelEncoder()
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    dfTrain = computeLoadedDataset(X_train, y_train)

    start_time = time.time()
    tree.dfTrain = dfTrain
    mpTrain, OriginalCandidatesListTrain, numberOfMotifTrain, numberOfDiscordTrain = getDataStructures(tree,
                                                                                                       dfTrain,
                                                                                                       tree.window_size,
                                                                                                       tree.k, verbose=False)

    # sfoltisco i candidati in base al gruppo di candidati scelti
    if (tree.candidatesGroup == 0):
        OriginalCandidatesListTrain = OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D'] == 0]
    if (tree.candidatesGroup == 1):
        OriginalCandidatesListTrain = OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D'] == 1]

    OriginalCandidatesListTrain.reset_index(drop=True)

    # aggiungo lista candidati e lista candidati usati, ORIGINALI, in tree
    tree.OriginalCandidatesUsedListTrain = buildCandidatesUsedList(OriginalCandidatesListTrain)
    tree.OriginalCandidatesListTrain = OriginalCandidatesListTrain
    if (verbose):
        print('OriginalCandidatesUsedListTrain: \n')
        print(tree.OriginalCandidatesUsedListTrain)

        print('OriginalCandidatesListTrain: \n')
        print(tree.OriginalCandidatesListTrain)

    # OriginalCandidatesListTrain VIENE MANTENUTO PER TUTTO L'ALGORITMO, DA ESSO AD OGNI SPLIT PRENDO CANDIDATI NECESSARI

    # PREPARO STRUTTURE DATI PER LA PRIMA ITERAZIONE DI FIT
    # applico clustering a insieme di candidati iniziali
    if (tree.useClustering):
        CandidatesListTrain = reduceNumberCandidates(tree, OriginalCandidatesListTrain, returnOnlyIndex=False)
        if (verbose):
            print('candidati rimasti/ più significativi-distintivi ')
    else:
        CandidatesListTrain = tree.OriginalCandidatesListTrain
    if (verbose):
        print(CandidatesListTrain)

    TsIndexList = dfTrain['TsIndex'].values  # inizialmente tutto DfTrain ( prima iterazione )

    # computeSubSeqDistance calcola distanze tra lista di Ts e lista di candidati fornite
    dfForDTree = computeSubSeqDistance(tree, TsIndexList, CandidatesListTrain, tree.window_size)
    if (verbose == True):
        print('dfTrain: \n' + str(dfTrain))
        print('dfForDTree: \n' + str(dfForDTree))


    #ESTRAGGO LA COLONNA LABEL
    #RIMUOVO COLONNA LABEL E TSINDEX INUTILI QUI
    y_train=dfForDTree['class']
    del dfForDTree["class"]
    del dfForDTree["TsIndex"]

    y_train = y_train.astype('int')

    print(dfForDTree)





    clf = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                                 min_samples_leaf=20)  # fissando random state ho sempre lo stesso valore e non ho ranodmicità nello split

    clf.fit(dfForDTree, y_train)  # genera DTree allenato su tr set


    #INIZIO STESSA PROCEDURA EPR GENERARE dfForDTreeTest
    dfTest = computeLoadedDataset(X_test, y_test)

    columns=dfForDTree.columns.values

    tree.attributeList=columns

    CandidatesListMatched = tree.OriginalCandidatesListTrain['IdCandidate'].isin(
        tree.attributeList)  # mi dice quali TsIndex in OriginalCandidatesListTrain sono contenuti in Dleft

    tree.dTreeAttributes = tree.OriginalCandidatesListTrain[
        CandidatesListMatched]

    dfForDTreeTest = computeSubSeqDistanceForTest(tree, dfTest, tree.dTreeAttributes )

    y_test=dfForDTreeTest["class"].values

    y_test=y_test.astype('int')

    del dfForDTreeTest["class"]
    print(dfForDTreeTest)

    y_predTest = clf.predict(dfForDTreeTest)

    timeToTest = time.time() - start_time

    print(classification_report(y_test, y_predTest))
    print('Accuracy %s' % accuracy_score(y_test, y_predTest))
    print('F1-score %s' % f1_score(y_test, y_predTest, average=None))
    confusion_matrix(y_test, y_predTest)

    row = ['Classic Classification', datasetName, accuracy_score(y_test, y_predTest), timeToTest]

    WriteCsvComparison('confrontoAlgoritmi.csv', row)


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



