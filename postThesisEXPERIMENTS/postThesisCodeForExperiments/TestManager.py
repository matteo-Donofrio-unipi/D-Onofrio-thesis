from Tree import *
from Tools import  *
from FileManager import *
import time
from sklearn.utils.random import sample_without_replacement
from tslearn.datasets import UCR_UEA_datasets
from pyts.transformation import ShapeletTransform
from PlotLibrary import plot_all, plotData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tslearn.shapelets import LearningShapelets, grabocka_params_to_shapelet_size_dict


#datasetNames = 'GunPoint,ItalyPowerDemand,ArrowHead,ECG200,ECG5000,PhalangesOutlinesCorrect'
def executeTestTSCMP(useValidationSet,usePercentageTrainingSet,datasetName,nameFile):#,initialWS,candidate):

    #INPUT: Parameters for TSCMP algorithm

    #Execution of a TSCMP test over the dataset: datasetName

    first = True  # Generation & Computation of the training dataset
    second = True  # Fittin of the Decision Tree
    third = True  # Generation & Computation of the testing dataset
    quarter = True  # Predict and show scores
    fifth=False   # Plot some/all classified instances
    sixth = False  # Plot of the choosen shapelet


    PercentageTrainingSet = 1 # variable percentage of the training set
    PercentageValidationSet = 0.3  # percentage of the training set chosen as validation set
    writeOnCsv = True


    le = LabelEncoder()
    tree= Tree(candidatesGroup=1,maxDepth=3,minSamplesLeaf=20,removeUsedCandidate=1,window_size=20,k=2,useClustering=True,n_clusters=20,warningDetected=False,verbose=0)



    if(first==True):
        verbose = False


        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)


        if(verbose):
            print('Initial Train set shape : ' + str(X_train.shape)+'\n')
            print('Initial Test set shape : ' + str(X_test.shape) + '\n')

        if(useValidationSet): #extract the validation set from the training set

            dimValidationSet = int(len(X_train) * PercentageValidationSet)
            selectedRecordsForValidation=sample_without_replacement(len(X_train), dimValidationSet)

            # get useful dataset
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


        if(usePercentageTrainingSet): #extract only a percentage from the training set

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


        # generate the TSCMP dataset from the oringial training dataset
        start_time = time.time()
        tree.dfTrain = dfTrain
        OriginalCandidatesListTrain, numberOfMotifTrain, numberOfDiscordTrain  = getDataStructures(tree,
            dfTrain, tree.window_size, tree.k, verbose)


        #select only the type of candidates chosen by the user
        if(tree.candidatesGroup==0):
            OriginalCandidatesListTrain=OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D']==0]
        if (tree.candidatesGroup == 1):
            OriginalCandidatesListTrain = OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D'] == 1]

        OriginalCandidatesListTrain.reset_index(drop=True)

        #add structures to tree
        tree.OriginalCandidatesUsedListTrain = buildCandidatesUsedList(OriginalCandidatesListTrain)
        tree.OriginalCandidatesListTrain=OriginalCandidatesListTrain
        if (verbose):
            print('OriginalCandidatesUsedListTrain: \n')
            print(tree.OriginalCandidatesUsedListTrain)

            print('OriginalCandidatesListTrain: \n')
            print(tree.OriginalCandidatesListTrain)


        #OriginalCandidatesListTrain remains the same during all the execution



        #prepare data srtucture for the execution by appliyng K-Medoids initially
        if(tree.useClustering):
            CandidatesListTrain = reduceNumberCandidates(tree, OriginalCandidatesListTrain,returnOnlyIndex=False)
            if (verbose):
                print('candidati rimasti/ più significativi-distintivi ')
        else:
            CandidatesListTrain=tree.OriginalCandidatesListTrain
        if (verbose):
            print(CandidatesListTrain)


        TsIndexList=dfTrain['TsIndex'].values #initially (first iteration) consider all the Ts

        # compute the euclidean dist btw each Ts and each chosen candidate
        dfForDTree = computeSubSeqDistance(tree,TsIndexList, CandidatesListTrain, tree.window_size)
        if (verbose == True):
            print('dfTrain: \n'+str(dfTrain))
            print('dfForDTree: \n'+str(dfForDTree))


    if(second==True):
        verbose = False
        #fit the Decision Tree
        tree.fit(dfForDTree,verbose=False)
        fitTime=time.time() - start_time #take the training phase time
        if(verbose==True):
            print(tree.attributeList)
            print(tree.Root)
            tree.printAll(tree.Root)

        print("--- %s seconds after fitting" % (fitTime))
        print(tree.SseList)
        print("SSE avg: ")
        print(sum(tree.SseList)/len(tree.SseList))



    if(third==True):
        verbose=False

        #Generate the test dataset
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
            tree.attributeList)  # set to true the candidate index chosen by the Decision Tree

        tree.dTreeAttributes = tree.OriginalCandidatesListTrain[
            CandidatesListMatched]  # extract the candidates chosen by the Decision Tree

        if(tree.verbose):
            print('Attributi selezionati dal Decision Tree')
            print(tree.dTreeAttributes)

        dfForDTreeTest=computeSubSeqDistanceForTest(tree,dfTest,tree.dTreeAttributes)
        if(tree.verbose==True):
            print(dfForDTreeTest)




    if(quarter==True):
        # Make prediction and show the results
        verbose = False

        #generate the dataset for plotting the classified instances
        tree.TsTestForPrint=list()
        temp=list()
        for i in range (len(dfTest)):
            temp = dfTest.iloc[i].values  # contiene la prima serie che viene classificata
            temp= temp[:len(temp) - 2]
            tree.TsTestForPrint.append(temp)
            temp=None


        yTest, yPredicted = tree.predict(dfForDTreeTest, tree.Root, fifth)

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

        #row=[datasetName,group,tree.maxDepth,tree.minSamplesLeaf,tree.window_size,tree.removeUsedCandidate,tree.k,useValidationSet,percentage,tree.useClustering,tree.n_clusters,round(aS,2),round(fitTime,2)]
        row = ['TSCMP', datasetName, round(aS,2),round(fitTime,2)]


        print('Classification Report  \n%s ' % cR)
        print('Accuracy %s' % aS)
        print('F1-score %s' % f1)
        print(" %s seconds END OF EXECUTION" % totalTime)

        #COMMENTO PER STAMPARE SU CONFRONTO ALGO
        if(writeOnCsv):
            #WriteCsv("TSCMPaperTest.csv", row)
            WriteCsvComparison('experimentsForPaper2.csv', row)

    if(sixth==True):

        #extract and plot the shapelet chosen by the Decision Tree
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
    # INPUT: Dataset name

    # Execution of a ShapeletTransformation algorithm over the dataset: datasetName

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
    #NB: IN ORDER TO MAKE A VALID COMPARISON WITH TSCMP, THE WINDOW SIZE VALUE MUST BE THE SAME OF THE VALUE CHOSEN IN TSCMP
    st = ShapeletTransform(window_sizes=[20],
                           random_state=41, sort=True)
    X_new = st.fit_transform(dfTrain, y_train)
    X_test_new = st.transform(dfTest)

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier()
    clf.fit(X_new, y_train)

    timeToFit = time.time() - start_time

    print("--- %s seconds after fitting" % (timeToFit)) #training phase time

    y_pred = clf.predict(X_test_new)

    timeToTest = time.time() - start_time

    print(accuracy_score(y_pred, y_test))

    print("--- %s seconds after testing" % (timeToTest))

    row = ['ShapeletTransformation', datasetName, round(accuracy_score(y_test, y_pred), 2), round(timeToFit, 2)]

    WriteCsvComparison('experimentsForPaper2.csv', row)


def executeLearningShapelet(datasetName):
    # INPUT: Dataset name

    # Execution of a ShapeletTransformation algorithm over the dataset: datasetName

    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    # RE-SIZE BY FUN X TRAIN
    dfTrain = computeLoadedDataset(X_train, y_train)

    y_train = dfTrain['target'].values
    y_train = y_train.astype(int)

    #get the number of classes
    le = LabelEncoder()
    distinct_classes = le.fit_transform(dfTrain['target'])
    distinct_classes = np.unique(distinct_classes, return_counts=False)
    num_classes = len(distinct_classes)

    print(distinct_classes)
    print(num_classes)

    del dfTrain['target']
    del dfTrain['TsIndex']




    # RE-SIZE BY FUN X TEST
    dfTest = computeLoadedDataset(X_test, y_test)

    y_test = dfTest['target'].values
    y_test = y_test.astype(int)

    del dfTest['target']
    del dfTest['TsIndex']

    start_time = time.time()

    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=len(dfTrain),
                                                           ts_sz=len(dfTrain.iloc[0]),
                                                           n_classes=num_classes,
                                                           l=0.1,  # parametri fissi
                                                           r=1)

    grabocka = LearningShapelets(n_shapelets_per_size=shapelet_sizes)
    grabocka.fit(dfTrain, y_train)
    X_train_distances = grabocka.transform(dfTrain)
    X_test_distances = grabocka.transform(dfTest)

    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_train_distances, y_train)
    y_predict = dt.predict(X_test_distances)

    accuracy_score(y_test, y_predict)



def executeClassicDtree(datasetName):
    # INPUT: Dataset

    # Execution of the DecisionTreeClassifier algorithm over the dataset: datasetName


    # NB: IN ORDER TO MAKE A VALID COMPARISON WITH TSCMP, THESE VALUES OF THE PARAMETERS MUST BE THE SAME OF THE VALUE CHOSEN IN TSCMP
    tree= Tree(candidatesGroup=1,maxDepth=3,minSamplesLeaf=20,removeUsedCandidate=1,window_size=20,k=2,useClustering=True,n_clusters=20,warningDetected=False,verbose=0)

    verbose=False

    #SAME INITIALIZATION AND DATA STRUCTURE GENERATION OF TSCMP

    le = LabelEncoder()
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    dfTrain = computeLoadedDataset(X_train, y_train)

    start_time = time.time()
    tree.dfTrain = dfTrain
    OriginalCandidatesListTrain, numberOfMotifTrain, numberOfDiscordTrain = getDataStructures(tree,
                                                                                                       dfTrain,
                                                                                                       tree.window_size,
                                                                                                       tree.k, verbose=False)


    if (tree.candidatesGroup == 0):
        OriginalCandidatesListTrain = OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D'] == 0]
    if (tree.candidatesGroup == 1):
        OriginalCandidatesListTrain = OriginalCandidatesListTrain[OriginalCandidatesListTrain['M/D'] == 1]

    OriginalCandidatesListTrain.reset_index(drop=True)

    tree.OriginalCandidatesUsedListTrain = buildCandidatesUsedList(OriginalCandidatesListTrain)
    tree.OriginalCandidatesListTrain = OriginalCandidatesListTrain
    if (verbose):
        print('OriginalCandidatesUsedListTrain: \n')
        print(tree.OriginalCandidatesUsedListTrain)

        print('OriginalCandidatesListTrain: \n')
        print(tree.OriginalCandidatesListTrain)


    if (tree.useClustering):
        CandidatesListTrain = reduceNumberCandidates(tree, OriginalCandidatesListTrain, returnOnlyIndex=False)
        if (verbose):
            print('candidati rimasti/ più significativi-distintivi ')
    else:
        CandidatesListTrain = tree.OriginalCandidatesListTrain

    if (verbose):
        print(CandidatesListTrain)

    TsIndexList = dfTrain['TsIndex'].values


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

    #print(dfForDTree)

    # NB: IN ORDER TO MAKE A VALID COMPARISON WITH TSCMP, THESE VALUES OF THE PARAMETERS MUST BE THE SAME OF THE VALUE CHOSEN IN TSCMP

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                                 min_samples_leaf=20)  # fissando random state ho sempre lo stesso valore e non ho ranodmicità nello split

    clf.fit(dfForDTree, y_train)

    timeToFit = time.time() - start_time #Training phase time

    print('Time to fit '+str(timeToFit))

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

    row = ['Decision Tree with Shapelet', datasetName, round(accuracy_score( y_test, y_predTest), 2), round(timeToFit, 2)]

    WriteCsvComparison('experimentsForPaper2.csv', row)




def executeDecisionTreeStandard(datasetName):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    #pre processing phase
    dfTrain = computeLoadedDataset(X_train, y_train)
    del dfTrain["TsIndex"]
    del dfTrain["target"]

    print(dfTrain)

    dfTest = computeLoadedDataset(X_test, y_test)


    y_test = y_test.astype('int')

    del dfTest["target"]
    del dfTest["TsIndex"]
    print(dfTest)

    #test phase
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                                 min_samples_leaf=20)
    start_time = time.time()

    clf.fit(dfTrain, y_train)

    timeToFit = time.time() - start_time  # Training phase time


    y_predTest = clf.predict(dfTest)

    print(classification_report(y_test, y_predTest))
    print('Accuracy %s' % accuracy_score(y_test, y_predTest))
    print('F1-score %s' % f1_score(y_test, y_predTest, average=None))
    confusion_matrix(y_test, y_predTest)

    row = ['Decision tree classifier', datasetName, round(accuracy_score(y_test, y_predTest),2), round(timeToFit,2)]

    WriteCsvComparison('experimentsForPaper2.csv', row)



def executeKNN(datasetName):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    scalerMM = MinMaxScaler()

    scalerS = StandardScaler()

    scalerUsed=1

    if(scalerUsed==1):
        scaler=scalerMM
    else:
        scaler=scalerS

    K=3

    # pre processing phase Training set
    dfTrain = computeLoadedDataset(X_train, y_train)
    del dfTrain["TsIndex"]
    del dfTrain["target"]

    dfTrain[dfTrain.columns] = scaler.fit_transform(dfTrain)

    print(dfTrain)

    # pre processing phase Test set
    dfTest = computeLoadedDataset(X_test, y_test)

    y_test = y_test.astype('int')

    del dfTest["target"]
    del dfTest["TsIndex"]

    dfTest[dfTest.columns] = scaler.fit_transform(dfTest)

    print(dfTest)

    # test phase

    knn = KNeighborsClassifier(n_neighbors=K)

    start_time = time.time()

    knn.fit(dfTrain, y_train)

    timeToFit = time.time() - start_time  # Training phase time
    # prediction on the test test
    test_pred_knn = knn.predict(dfTest)



    print(classification_report(y_test, test_pred_knn))
    print('Accuracy %s' % accuracy_score(y_test, test_pred_knn))
    print('F1-score %s' % f1_score(y_test, test_pred_knn, average=None))
    confusion_matrix(y_test, test_pred_knn)


    row = ['KNN', datasetName, round(accuracy_score(y_test, test_pred_knn),2), round(timeToFit,2)]

    WriteCsvComparison('experimentsForPaper2.csv', row)

