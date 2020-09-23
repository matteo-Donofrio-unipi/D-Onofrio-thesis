from Tree import *
from PreProcessingLibrary import  *
from TestFileManager import *
import time
from sklearn.utils.random import sample_without_replacement
from tslearn.datasets import UCR_UEA_datasets
from pathlib import Path
from datetime import datetime

first=False #ESTRAZIONE DATASET TRAINING
second=False #CALCOLO ALBERO DECISIONE
third=False #ESTRAZIONE DATASET TEST
quarter=False #PREDIZIONE E RISULTATO
fifth=False #GRAFICA DELLE SERIE TEMPORALI
sixth=False #GRAFICA DI SERIE TEMPORALI E MATRIX PROFILE DEI CANDIDATI SCELTI
seventh=True

#genero albero (VUOTO) e avvio timer
tree= Tree(candidatesGroup=1,maxDepth=4,minSamplesLeaf=15,removeUsedCandidate=0,window_size=10,k=1,verbose=1) # K= NUM DI MOTIF/DISCORD ESTRATTI
datasetName='ItalyPowerDemand'
nameFile = datasetName + 'TestResults.csv'
start_time = time.time()




if(first==True):
    #ACQUISISCO STRUTTURE DATI DEL TRAINING SET
    verbose = True

    #CARICO DATI DAL FILE
    #datasetTrain = arff.loadarff('ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff')
    #dfTrain=pd.DataFrame(datasetTrain[0])

    #CARICO DATI DA LIBRERIA (FUNZIONE)
    le = LabelEncoder()


    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    dimWholeTrainSet=len(X_train)
    PercentageTrainingSet=0.1
    dimSubTrainSet= int(len(X_train) * PercentageTrainingSet) #dim of new SubSet of X_train

    selectedRecords=sample_without_replacement(len(X_train), dimSubTrainSet) #random records selected
    print(selectedRecords)
    X_train=X_train[selectedRecords]
    #print(X_train)


    dfTrain = computeLoadedDataset(X_train, y_train)
    num_classes = le.fit_transform(dfTrain['target'])
    print(np.unique(num_classes, return_counts=True))
    patternLenght=len(dfTrain.iloc[0])-1
    print('PATT LENGHT: '+str(patternLenght))

    # genero strtutture dati ausiliarie

    mpTrain,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain,CandidatesUsedListTrain=getDataStructures(dfTrain,tree.window_size,tree.k,verbose=1)
    dfForDTree = computeSubSeqDistance(dfTrain, CandidatesListTrain, tree.window_size)
    if(verbose==True):
        print(dfTrain)
        print(dfForDTree)

    print("--- %s seconds after getting DATA STRUCTURES" % (time.time() - start_time))


if(second==True):
    verbose = True
    #COSTRUISCO DECISION TREE
    #------
    tree.fit(dfForDTree,CandidatesUsedListTrain,numberOfMotifTrain,numberOfDiscordTrain,verbose=True)
    if(verbose==True):
        print(tree.attributeList)
        print(tree.Root)
        tree.printAll(tree.Root)

    print("--- %s seconds after building TREE" % (time.time() - start_time))



if(third==True):
    #GENERO STRUTTURE DATI PER TEST SET
    verbose=True

    # CARICO DATI DAL FILE
    #datasetTest = arff.loadarff('ItalyPowerDemand/ItalyPowerDemand_TEST.arff')
    #dfTest = pd.DataFrame(datasetTest[0])  # 30 record su matrice da 128 attributi + 'b': classe appartenenza


    # CARICO DATI DA LIBRERIA (FUNZIONE)
    dfTest = computeLoadedDataset(X_test, y_test)
    #dfTest = dfTest.iloc[10:20]  # ne prendo 50 altrimenti impiega tempo troppo lungo, sono 900 record totali


    tree.attributeList=sorted(tree.attributeList) #ordino attributi per rendere più efficiente 'computeSubSeqDistanceForTest'
    tree.attributeList=np.unique(tree.attributeList)
    dfForDTreeTest,TsAndStartingPositionList=computeSubSeqDistanceForTest(dfTest,dfTrain,tree.attributeList,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain,tree.window_size)
    if(verbose==True):
        print(dfForDTreeTest)
        print(TsAndStartingPositionList)




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

    row=[group,tree.maxDepth,tree.minSamplesLeaf,tree.window_size,tree.removeUsedCandidate,tree.k,PercentageTrainingSet,round(aS,2),round(totalTime,2)]
    print('Classification Report %s' % cR)
    print('Accuracy %s' % aS)
    print('F1-score %s' % f1)
    print(" %s seconds END OF EXECUTION" % totalTime)

    #salvo su file i risultati di questa configuazione
    #Path("./TestResults").mkdir(parents=True, exist_ok=True)


    WriteCsv(nameFile, row)

    # file = open(nameFile, "a+")
    #
    #
    # file.write('DATASET : %s   \n' % datasetName)
    # file.write('HYPER-PARAMETER-VALUES   \n')
    # file.write('Candidates Group :  %s  \nMax Depth Tree :  %d  \nMin Samples for Leaf :  %d  \nUsed Candidates are removed :  %d  \nWindow Size :  %d  \nK (# motif/discord retrieved) :  %d  \n  \n' % (group,tree.maxDepth,tree.minSamplesLeaf,tree.removeUsedCandidate,tree.window_size,tree.k))
    # file.write('\nDATASET INFO   \n')
    # file.write('#Pattern Training :  %d  \n#Pattern Training used :  %d  \n#Pattern Test :  %d  \nLength pattern :  %d  \n' % ( dimWholeTrainSet, dimSubTrainSet,len(dfTest), patternLenght) )
    # file.write('CLASSIFICATION REPORT:\n %s \nAccuracy %s \nF1-score %s \n' % (cR , aS ,f1))
    # file.write("%s seconds END OF EXECUTION \n " % str(totalTime))
    # file.write('\n \n \n ---------------------------------------------------- \n \n \n' )
    # file.close()


if(fifth==True):

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






if(sixth==True):
    #ESTRAGGO TUTTO DI NUOVO PERCHE LE TS USATE PRIMA HANNO VALORI AGGIUNTI
    diz = {'Motif': [], 'Discord': []}
    PreProcessedTs = pd.DataFrame(diz)

    for i in range(len(dfTrain)):
        for j in range(len(TsAndStartingPositionList)):
            if(TsAndStartingPositionList[j][0] == i):
                Ts = np.array(dfTrain.iloc[i].values)
                print('TS ID:' + str(i))
                print('TS CLASS:' + str(dfTrain.iloc[i]['target']))
                mp, mot, motif_dist, dis = retrieve_all2(Ts,tree.window_size,tree.k)
                print('CANDIDATE START IN : '+str(TsAndStartingPositionList[j][1]))
                plot_all(Ts, mp, mot, motif_dist, dis, tree.window_size)
                PreProcessedTs.loc[i] = mot, dis


if(seventh==True):
    PlotValues(nameFile)


