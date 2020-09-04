# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from DecisionTreeClassifier import *
from plotFun import *
import time

# Press the green button in the gutter to run the script.
first=False
second=False
third=False
quarter=False
fifth=True


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
    setVariables(5,attributeList)
    verbose=True
    mpTrain,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain,CandidatesUsedListTrain=getDataStructures(dfTrain,verbose)
    print(dfTrain)
    dfForDTree=computeSubSeqDistance(dfTrain,CandidatesListTrain)
    if(verbose==True):
        print(dfForDTree)

    print("--- %s seconds after getting DATA STRUCTURES" % (time.time() - start_time))


if(second==True):
    #COSTRUISCO DECISION TREE
    candidatesGroup=0
    albero=None
    maxDepth=3
    minSamplesLeaf=5
    removeUsedCandidate=1
    verbose=True
    albero=fit(dfForDTree[:30],candidatesGroup,CandidatesUsedListTrain,maxDepth,minSamplesLeaf,numberOfMotifTrain,numberOfDiscordTrain,removeUsedCandidate,verbose)
    print(attributeList)
    print(albero)
    printAll(albero)

    print("--- %s seconds after building TREE" % (time.time() - start_time))


if(third==True):
    #GENERO STRUTTURE DATI PER TEST SET
    verbose=True
    dataset2 = arff.loadarff('ItalyPowerDemand/ItalyPowerDemand_TEST.arff')
    dfTest = pd.DataFrame(dataset2[0]) #30 record su matrice da 128 attributi + 'b': classe appartenenza
    dfTest=dfTest.iloc[:60] #ne prendo 50 altrimenti impiega tempo troppo lungo, sono 900 record totali

    attributeList=sorted(attributeList) #ordino attributi per rendere più efficiente 'computeSubSeqDistanceForTest'
    dfForDTreeTest=computeSubSeqDistanceForTest(dfTest,dfTrain,attributeList,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain)
    if(verbose==True):
        print(dfForDTreeTest)


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

    print("--- %s seconds END OF EXECUTION" % (time.time() - start_time)/60)


if(fifth==True):
    window_size = 5
    setVariablesForPlot(window_size)
    dataset2 = arff.loadarff('CBF/CBF_TRAIN.arff')
    dfTrain2 = pd.DataFrame(dataset2[0])
    diz = {'Motif': [], 'Discord': []}
    PreProcessedTs = pd.DataFrame(diz)

    for i in range(25):
        Ts = np.array(dfTrain2.iloc[i].values)
        print('TS ID:' + str(i))
        print('TS CLASS:' + str(Ts[len(Ts)-1]))
        mp, mot, motif_dist, dis = retrieve_all2(Ts)
        # print('MP'+str(mp))
        print("Motifs" + str(mot))
        print("Motifs Dist" + str(motif_dist))
        print("Discords" + str(dis))
        plot_all(Ts, mp, mot, motif_dist, dis, window_size)
        PreProcessedTs.loc[i] = mot, dis

    PreProcessedTs = candidateFilter2(PreProcessedTs)
    print('Motif/Discord estratti')
    print(PreProcessedTs)
