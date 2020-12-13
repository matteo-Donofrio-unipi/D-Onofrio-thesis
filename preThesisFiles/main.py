from Tree import *
from PreProcessingLibrary import  *
import time
from tslearn.datasets import UCR_UEA_datasets
from pathlib import Path
from datetime import datetime

first=True #ESTRAZIONE DATASET TRAINING
second=True #CALCOLO ALBERO DECISIONE
third=True #ESTRAZIONE DATASET TEST
quarter=True #PREDIZIONE E RISULTATO
fifth=False #GRAFICA DELLE SERIE TEMPORALI

#genero albero (VUOTO) e avvio timer
tree= Tree(candidatesGroup=1,maxDepth=3,minSamplesLeaf=5,removeUsedCandidate=0,verbose=1)
start_time = time.time()






if(first==True):
    #ACQUISISCO STRUTTURE DATI DEL TRAINING SET
    verbose = True

    #CARICO DATI DAL FILE
    #datasetTrain = arff.loadarff('ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff')
    #dfTrain=pd.DataFrame(datasetTrain[0])

    #CARICO DATI DA LIBRERIA (FUNZIONE)
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('CBF')
    dfTrain = computeLoadedDataset(X_train, y_train)

    # genero strtutture dati ausiliarie
    window_size = 5
    mpTrain,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain,CandidatesUsedListTrain=getDataStructures(dfTrain,window_size,verbose=1)
    dfForDTree = computeSubSeqDistance(dfTrain, CandidatesListTrain, window_size)
    if(verbose==True):
        print(dfForDTree)

    print("--- %s seconds after getting DATA STRUCTURES" % (time.time() - start_time))


if(second==True):
    verbose = True
    #COSTRUISCO DECISION TREE
    #------
    tree.fit(dfForDTree[:5],CandidatesUsedListTrain,numberOfMotifTrain,numberOfDiscordTrain,verbose=True)
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
    dfTest = dfTest.iloc[10:20]  # ne prendo 50 altrimenti impiega tempo troppo lungo, sono 900 record totali


    tree.attributeList=sorted(tree.attributeList) #ordino attributi per rendere più efficiente 'computeSubSeqDistanceForTest'
    dfForDTreeTest,TsAndStartingPositionList=computeSubSeqDistanceForTest(dfTest,dfTrain,tree.attributeList,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain,window_size)
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

    print('Classification Report %s' % cR)
    print('Accuracy %s' % aS)
    print('F1-score %s' % f1)
    print(" %s seconds END OF EXECUTION" % totalTime)

    #salvo su file i risultati di questa configuazione
    Path("./TestResults").mkdir(parents=True, exist_ok=True)
    dateTimeObj = datetime.now()
    nameFile='./TestResults/TestResults1.txt'
    file = open(nameFile, "a+")

    if(tree.candidatesGroup==0):
        group='Motifs'
    elif(tree.candidatesGroup == 1):
        group = 'Discords'
    else:
        group = 'Both'

    file.write('Hyper-parameter value : \n')
    file.write('Candidates Group :  %s  \nMax Depth Tree :  %d  \nMin Samples for Leaf :  %d  \nUsed Candidates are removed :  %d  \n  \n' % (group,tree.maxDepth,tree.minSamplesLeaf,tree.removeUsedCandidate))
    file.write('Classification Report:\n %s \nAccuracy %s \nF1-score %s \n' % (cR , aS ,f1))
    file.write("%s seconds END OF EXECUTION \n " % str(totalTime))
    file.write('\n \n \n ---------------------------------------------------- \n \n \n' )
    file.close()



if(fifth==True):
    #ESTRAGGO TUTTO DI NUOVO PERCHE LE TS USATE PRIMA HANNO VALORI AGGIUNTI
    window_size = 5
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
                mp, mot, motif_dist, dis = retrieve_all2(Ts,window_size)
                print('CANDIDATE START IN : '+str(TsAndStartingPositionList[j][1]))
                plot_all(Ts, mp, mot, motif_dist, dis, window_size)
                PreProcessedTs.loc[i] = mot, dis

