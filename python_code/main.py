from DecisionTreeClassifier import *
from plotFun import *
import time

# Boolean verbose
first=True #ESTRAZIONE DATASET TRAINING
second=True #CALCOLO ALBERO DECISIONE
third=True #ESTRAZIONE DATASET TEST
quarter=True #PREDIZIONE E RISULTATO
fifth=True #GRAFICA DELLE SERIE TEMPORALI


print('TEST CLASS')
stringFile0='CBF/CBF_TRAIN.arff'
stringFile1='ItalyPowerDemand/ItalyPowerDemand_TEST.arff'
start_time = time.time()

if(first==True):
    #ACQUISISCO STRUTTURE DATI DEL TRAINING SET
    dataset = arff.loadarff('ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff')
    dfTrain=pd.DataFrame(dataset[0])
    window_size=5
    attributeList=list()
    setVariables(window_size,attributeList)
    verbose=True
    mpTrain,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain,CandidatesUsedListTrain=getDataStructures(dfTrain,verbose)
    print(dfTrain)
    dfForDTree=computeSubSeqDistance(dfTrain,CandidatesListTrain)
    if(verbose==True):
        print(dfForDTree)

    print("--- %s seconds after getting DATA STRUCTURES" % (time.time() - start_time))


if(second==True):
    #COSTRUISCO DECISION TREE
    #SET IPER-PARAMETRI -----
    candidatesGroup=1
    maxDepth=3
    minSamplesLeaf=5
    removeUsedCandidate=0
    #------
    verbose=True
    albero=fit(dfForDTree[:5],candidatesGroup,CandidatesUsedListTrain,maxDepth,minSamplesLeaf,numberOfMotifTrain,numberOfDiscordTrain,removeUsedCandidate,verbose)
    print(attributeList)
    print(albero)
    printAll(albero)

    print("--- %s seconds after building TREE" % (time.time() - start_time))


if(third==True):
    #GENERO STRUTTURE DATI PER TEST SET
    verbose=True
    dataset2 = arff.loadarff('ItalyPowerDemand/ItalyPowerDemand_TEST.arff')
    dfTest = pd.DataFrame(dataset2[0]) #30 record su matrice da 128 attributi + 'b': classe appartenenza
    dfTest=dfTest.iloc[:10] #ne prendo 50 altrimenti impiega tempo troppo lungo, sono 900 record totali

    attributeList=sorted(attributeList) #ordino attributi per rendere più efficiente 'computeSubSeqDistanceForTest'
    dfForDTreeTest,TsAndStartingPositionList=computeSubSeqDistanceForTest(dfTest,dfTrain,attributeList,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain)
    if(verbose==True):
        print(dfForDTreeTest)
        print(TsAndStartingPositionList)

if(quarter==True):
    # EFFETTUO PREDIZIONE E MISURO RISULTATO
    verbose = True
    yTest, yPredicted = predict(dfForDTreeTest, albero)

    if (verbose == True):
        for a, b in zip(yTest, yPredicted):
            print(a, b)

    print(type(yPredicted))
    print(type(yTest))

    print(classification_report(yTest, yPredicted))
    print('Accuracy %s' % accuracy_score(yTest, yPredicted))
    print('F1-score %s' % f1_score(yTest, yPredicted, average=None))
    confusion_matrix(yTest, yPredicted)

    print("--- %s seconds END OF EXECUTION" % (time.time() - start_time))


if(fifth==True):
    #ESTRAGGO TUTTO DI NUOVO PERCHE LE TS USATE PRIMA HANNO VALORI AGGIUNTI
    window_size = 5
    setVariablesForPlot(window_size)
    dataset2 = arff.loadarff('ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff')
    dfTrain2 = pd.DataFrame(dataset2[0])
    diz = {'Motif': [], 'Discord': []}
    PreProcessedTs = pd.DataFrame(diz)

    for i in range(len(dfTrain2)):
        for j in range(len(TsAndStartingPositionList)):
            if(TsAndStartingPositionList[j][0] == i):
                Ts = np.array(dfTrain2.iloc[i].values)
                print('TS ID:' + str(i))
                print('TS CLASS:' + str(Ts[len(Ts)-1]))
                mp, mot, motif_dist, dis = retrieve_all2(Ts)
                print('CANDIDATE START IN : '+str(TsAndStartingPositionList[j][1]))
                plot_all(Ts, mp, mot, motif_dist, dis, window_size)
                PreProcessedTs.loc[i] = mot, dis

