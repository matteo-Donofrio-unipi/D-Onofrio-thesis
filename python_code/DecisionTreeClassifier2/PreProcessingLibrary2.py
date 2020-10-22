import pandas as pd
import numpy as np
from matrixprofile import *
from matrixprofile.discords import discords
from matplotlib import pyplot as plt
from scipy.io import arff
from binarytree import Node
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
from math import log, e
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
import math
from scipy.spatial.distance import euclidean
from kCandidatesSearch2 import runKMeans


# LIBRERIA PER TUTTE LE FUNZIONI DI PRE PROCESSING DEI DATI
# DALL'ESTRAZIONE AL PASSAGGIO PER LA CREAZIONE DELL'ALBERO



def computeLoadedDataset(X, y):

    columnsList = np.arange(len(X[0]))
    columnsList2 = list()
    prefix = 'att'

    #conto il numero di attributi e genero nomi colonne
    for i in columnsList:
        columnsList2.append(prefix + str(i))
    columnsList2.append('target')
    columnsList2.append('TsIndex')
    dataset = pd.DataFrame(columns=columnsList2, index=range(0, len(X)))

    #aggiungo ad ogni record l'attributo classe
    for i in range(len(X)):
        l1 = list()
        record = X[i]
        for j in range(len(X[i])):
            l1.append(record[j][0])
        l1.append(y[i])
        l1.append(i)
        dataset.iloc[i] = l1

    return dataset

def retrieve_all(tree,Ts,window_size,k):  # fornita la Ts calcola e restituisce mp, motifs, motifs_distances e discords
    dfMP = pd.DataFrame(Ts).astype(float)  # genero Dframe per lavorarci su, DA CAPIRE PERCHE SERVE FLOAT
    dis=[]


    if(tree.warningDetected==True):
        mp, mpi = matrixProfile.naiveMP(dfMP[0].values, window_size)
    else:
        mp, mpi = matrixProfile.stomp(dfMP[0].values, window_size)

    if(np.isnan(mp).any() or np.isinf(mp).any()):
        tree.warningDetected=True
        print('switch to ComputeMpAndMpi')
        mp, mpi = matrixProfile.naiveMP(dfMP[0].values, window_size)

    # PREPARO TUPLA DA PASSARE ALLA FUN MOTIF (RICHIEDE TUPLA FATTA DA MP E MPI)
    tupla = mp, mpi

    mot, motif_dist = motifs.motifs(dfMP[0].values, tupla, k)

    # CALCOLO DISCORDS
    if(sum(mp)!=0):
        dis = discords(mp, window_size, k)
    # print('Discords starting position: '+str(dis))

    tupla = mp, mot, motif_dist, dis
    return tupla




#my function for compute mp and mpi
def ComputeMpAndMpi(Ts, window_size):
    if window_size >= len(Ts) or window_size < 2:
        raise ValueError('Window_size not supported')

    Ts = Ts.astype(float)
    lenTs = len(Ts)
    mp = list()
    mpi = list()

    for i in range(lenTs):

        bestDist = 1000
        bestIdx = 0

        if (i + window_size > lenTs):
            break
        else:
            subSeq = Ts[i:i + window_size]

            for j in range(lenTs):
                if(i==j):
                    continue
                if (j + window_size > lenTs):
                    break
                else:
                    subSeqToCompute = Ts[j:j + window_size]
                    dist=euclidean(subSeq,subSeqToCompute)

                    if (dist > 0 and dist < bestDist):
                        bestDist = dist
                        bestIdx = j  # starting index of founded closest subseq

            mp.append(bestDist)
            mpi.append(bestIdx)

    return mp, mpi

#my fun for compute the Distance Profile
#CALCOLA DP TRA SUBSEQ A CONTENUTA IN TsContainigSubSeq E TUTTE LE SUBSEQ B CONTENUTE IN TsToCompare
def ComputeDp(TsContainigSubSeq, indexStartigPosition, window_size, TsToCompare=None):
    TsContainigSubSeq = TsContainigSubSeq.astype(float)
    ComparingWithItSelf = False

    #controllo e inizializzola TS su cui calcolare DP
    if (TsToCompare is None):
        ComparingWithItSelf = True
        TsToCompare = TsContainigSubSeq
    else:
        TsToCompare = TsToCompare.astype(float)

    if window_size >= len(TsToCompare) or window_size < 2:
        raise ValueError('Window_size not supported')

    #subSeq A
    subSeq = TsContainigSubSeq[indexStartigPosition:indexStartigPosition + window_size]

    lenTs = len(TsToCompare)
    dp = list()

    for i in range(lenTs):

        if (i + window_size > lenTs):
            break
        elif (i == indexStartigPosition and ComparingWithItSelf == True):
            continue
        else:
            # subSeq A generata in ogni iterazione
            subSeqToCompute = TsToCompare[i:i + window_size]
            dist = euclidean(subSeq, subSeqToCompute)
            dp.append(dist)

    return dp



# ogni motif e identificato da almeno due indici di partenza nella Ts, ne prendo uno solo rappresentativo
# genero poi struttura contenente gli indici di partenza di tutti i candidati

def candidateFilter(CandidatesList):
    counterNumberMotif = 0
    counterNumberDiscord = 0
    l2 = np.array([])
    for i in range(len(CandidatesList['Motif'])):  # per ogni entry (per ogni record)
        numMotif = len(CandidatesList['Motif'].iloc[i])
        numDiscord = len(CandidatesList['Discord'].iloc[i])
        counterNumberDiscord += numDiscord
        for j in range(numMotif):  # per ogni lista di motif
            l1 = CandidatesList['Motif'].iloc[i]  # prima lista
            l2 = np.append(l2, l1[j][0])  # prendo primo valore di ogni lista
            counterNumberMotif += 1

        CandidatesList['Motif'].iloc[i] = l2
        l2 = np.array([])  # svuoto array

    return CandidatesList, counterNumberMotif, counterNumberDiscord



#conto dopo aver selezionato i medoidi, il num di motif e discords
def countNumberOfCandidates(CandidatesListTrain):

    numMotifs=0
    numDiscords=0

    numMotifs=len(CandidatesListTrain[CandidatesListTrain['M/D']==0])
    numDiscords = len(CandidatesListTrain[CandidatesListTrain['M/D'] == 1])

    return numMotifs,numDiscords





# riceve la lista di candidati e genera una lista della stessa dimensione di booleani
def buildCandidatesUsedList(CandidatesList):
    CandidatesUsedList = pd.DataFrame(columns=['IdCandidate','Used'], index=range(len(CandidatesList)))
    boolList = [False] * (len(CandidatesList))
    CandidatesUsedList['Used'] = boolList
    CandidatesUsedList['IdCandidate']=CandidatesList['IdCandidate'].values
    return CandidatesUsedList



def getDataStructures(tree,df,window_size,k,verbose):
    # trasformo da stringa a numero il campo target
    le = LabelEncoder()
    num_classes = le.fit_transform(df['target'])
    df['target'] = num_classes


    diz = {'IdTs':[],'IdCandidate':[],'startingPosition':[],'M/D':[]}
    numberOfMotifs=0
    numberOfDiscords=0

    # CALCOLO MOTIF E DISCORD E LI INSERISCO NEL DIZIONARIO

    if(verbose):
        print('start computing MP, MPI')

    counter=0 #incremental counter for candidates
    for i in range(len(df)):
        if(tree.warningDetected==True and i % 100 == 0):
            print('computing Ts #: '+str(i))

        Ts = np.array(df.iloc[i][:-2].values) #-2 perche rimuovo l'attributo target e index inserito precedentemente

        tupla= retrieve_all(tree,Ts,window_size,k)
        mp, mot, motif_dist, dis = tupla



        for j in range(len(mot)):
            diz['IdTs'].insert(counter,df.iloc[i]['TsIndex'])
            diz['IdCandidate'].insert(counter, counter)
            diz['startingPosition'].insert(counter, mot[j][0])
            diz['M/D'].insert(counter, 0)
            numberOfMotifs +=1
            counter+=1

        for j in range(len(dis)):
            diz['IdTs'].insert(counter, df.iloc[i]['TsIndex'])
            diz['IdCandidate'].insert(counter, counter)
            diz['startingPosition'].insert(counter, dis[j])
            diz['M/D'].insert(counter, 1)
            numberOfDiscords +=1
            counter+=1


    # GENERO DFRAME DA DIZIONARIO
    CandidatesList = pd.DataFrame(diz)
    if (verbose == True):
        print('Candidati estratti: ')
        print(CandidatesList)
        print('numberOfMotif: %d, numberOfDiscord: %d \n'% (numberOfMotifs, numberOfDiscords))
        print('\n')

    return mp, CandidatesList, numberOfMotifs, numberOfDiscords



def computeSubSeqDistance(tree,EntireDataset, TsIndexList ,CandidatesList,window_size):

    columnsList = CandidatesList['IdCandidate']
    columnsList2 = list()
    lastAttribute = ['TsIndex', 'class']
    prefix = 'cand'
    for i in columnsList:
        columnsList2.append(prefix + str(i))
    columnsList2.append('TsIndex')
    columnsList2.append('class')
    dfForDTree = pd.DataFrame(columns=columnsList2, index=range(0, len(TsIndexList)))

    print('NEL LOOP')
    for i in range(len(TsIndexList)):
        # acquisisco la Ts di cui calcolare distanza
        TsIndexValue=TsIndexList[i]

        TsToCompare=np.array(EntireDataset[EntireDataset['TsIndex']==TsIndexValue].values)

        TsToCompare=TsToCompare[0]
        classValue = TsToCompare[len(TsToCompare)-2] #la classe è sempre il penultimo attributo
        TsToCompare = TsToCompare[:len(TsToCompare)-3] #la serie è ottenuta rimuovendo i due ultimi attributi

        dfForDTree['TsIndex'].iloc[i] = TsIndexValue
        dfForDTree['class'].iloc[i] = classValue

        for j in range (len(CandidatesList)):
            IdCandidate=CandidatesList.iloc[j]['IdCandidate']
            IdTsCandidate=CandidatesList.iloc[j]['IdTs']
            startingPosition=CandidatesList.iloc[j]['startingPosition']

            TsContainingCandidate = np.array(EntireDataset[EntireDataset['TsIndex'] == IdTsCandidate].values)
            TsContainingCandidate=TsContainingCandidate[0]
            TsContainingCandidate = TsContainingCandidate[:len(TsContainingCandidate) - 3]

            if (tree.warningDetected):
                Dp = distanceProfile.naiveDistanceProfile(TsContainingCandidate, int(startingPosition),
                                                          window_size, TsToCompare)
            else:
                Dp = distanceProfile.massDistanceProfile(TsContainingCandidate, int(startingPosition),
                                                         window_size, TsToCompare)
            minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
            dfForDTree[prefix + str(IdCandidate)].iloc[i] = minValueFromDProfile


    return dfForDTree


# per ogni Ts calcolo Dprofile con ogni candidato e inserisco la distanza minima con candidato i-esimo nella colonna i-esima
def computeSubSeqDistance2(tree,dataset, CandidatesList,window_size,numberOfCandidates):
    print('start computing df for dtree')

    # quantifico il num di candidati e in base a tale valore genero colonne per dfForDTree

    columnsList = np.arange(numberOfCandidates)
    columnsList2 = list()
    lastAttribute = ['TsIndex', 'class']
    prefix = 'cand'
    for i in columnsList:
        columnsList2.append(prefix + str(i))
    columnsList2.append('TsIndex')
    columnsList2.append('class')
    dfForDTree = pd.DataFrame(columns=columnsList2, index=range(0, len(dataset)))

    # per ogni Ts, scandisco ogni candidato e calcolo la distanza minore
    for i in range(len(dataset)):
        # acquisisco la Ts
        TsToCompare = np.array(dataset.iloc[i].values)
        classValue = TsToCompare[len(TsToCompare)-2] #la classe è sempre il penultimo attributo
        TsToCompare = TsToCompare[:len(TsToCompare)-3] #la serie è ottenuta rimuovendo i due ultimi attributi
        dfForDTree['TsIndex'].iloc[i] = i
        dfForDTree['class'].iloc[i] = classValue
        counter = 0
        # raccolgo i candidati di una riga
        for index, row in CandidatesList.iterrows():
            numMotif = len(row['Motif'])
            # scandisco e calcolo distanza dai motif
            for k in range(numMotif):
                l1 = row['Motif']  # lista di indice i in motifDiscordList
                startingIndex = l1[k]  # indice di inizio del motif
                TsContainingCandidateShapelet = np.array(dataset.iloc[index].values)  # Ts contenente candidato shapelet JCONTROLLARE
                if(tree.warningDetected):
                    Dp = distanceProfile.naiveDistanceProfile(TsContainingCandidateShapelet, int(startingIndex), window_size,TsToCompare)
                else:
                    Dp = distanceProfile.massDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                              window_size, TsToCompare)
                minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
                dfForDTree[prefix + str(counter)].iloc[i] = minValueFromDProfile
                counter += 1

        for index, row in CandidatesList.iterrows():
            numDiscord = len(row['Discord'])
            # scandisco e calcolo distanza dai discord
            for k in range(numDiscord):
                l1 = row['Discord']  # lista di indice i in motifDiscordList
                startingIndex = l1[k]  # indice di inizio del motif
                TsContainingCandidateShapelet = np.array(dataset.iloc[index].values)  # Ts contenente candidato shapelet
                if (tree.warningDetected):
                    Dp = distanceProfile.naiveDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                              window_size, TsToCompare)
                else:
                    Dp = distanceProfile.massDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                             window_size, TsToCompare)
                minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
                dfForDTree[prefix + str(counter)].iloc[i] = minValueFromDProfile
                counter += 1

    # print(counter)
    return dfForDTree




def computeSubSeqDistanceForTest(tree,datasetTest, datasetTrain, attributeList, CandidatesList, numberOfMotif,
                                 numberOfDiscord,window_size):
    # quantifico il num di candidati usati dall'albero e in base a tale valore genero colonne per dfForDTree
    columnsList2 = list()
    prefix = 'cand'
    TsAndStartingPositionList=list() #contiene le coppie (Ts, startingPosition) per tenere traccia degli shaplet ottenuti
    booleanForTsAndStartingPos=True #dopo aver raccolto dati nel primo giro, lo setto a false e mi fermo (altrimenti raccolgo stessi dati)
    for i in attributeList:
        columnsList2.append(prefix + str(i))
    columnsList2.append('TsIndex')
    columnsList2.append('class')
    dfForDTreeTest = pd.DataFrame(columns=columnsList2, index=range(0, len(datasetTest)))

    # per ogni Ts, scandisco ogni candidato e calcolo la distanza minore
    for i in range(len(datasetTest)):
        # acquisisco la Ts
        TsToCompare = np.array(datasetTest.iloc[i].values)
        classValue = TsToCompare[len(TsToCompare) - 1]  # la classe è sempre il penultimo attributo
        TsToCompare = TsToCompare[:len(TsToCompare) - 2]  # la serie è ottenuta rimuovendo i due ultimi attributi
        #I VALORI (-1, -2) SONO DIVERSI DA QUELLI USATI IN COMPUTE NORMALE, PERCHE QUI NON PASSO LA STRUTTURA A GETDATASTRUCTURES => NON AGGIUNGO COLONNA TS INDEX
        dfForDTreeTest['TsIndex'].iloc[i] = i
        dfForDTreeTest['class'].iloc[i] = classValue
        counter = 0 #scandisco candidate list (prima motif poi discord) incrementando counter -> cosi prenderò il candidato counter-esimo
        # scandisco e calcolo distanza dai candidati
        for z in range(len(attributeList)):
            candidateIndex = attributeList[z]
            counter = 0
            for index, row in CandidatesList.iterrows():
                numMotif = len(row['Motif'])
                for k in range(numMotif):
                    if (counter == candidateIndex):
                        l1 = row['Motif']  # lista di indice i in motifDiscordList
                        startingIndex = l1[k]  # indice di inizio del motif
                        TsContainingCandidateShapelet = np.array(
                            datasetTrain.iloc[index].values)  # Ts contenente candidato shapelet
                        if (tree.warningDetected):
                            Dp = distanceProfile.naiveDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                                      window_size, TsToCompare)
                        else:
                            Dp = distanceProfile.massDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                                     window_size, TsToCompare)
                        minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
                        dfForDTreeTest[prefix + str(counter)].iloc[i] = minValueFromDProfile
                        if(booleanForTsAndStartingPos==True):
                            TsAndStartingPositionList.append([index,startingIndex])
                    counter += 1
            for index, row in CandidatesList.iterrows():
                numDiscord = len(row['Discord'])
                for k in range(numDiscord):
                    if (counter == candidateIndex):
                        l1 = row['Discord']  # lista di indice i in motifDiscordList
                        startingIndex = l1[k]  # indice di inizio del motif
                        TsContainingCandidateShapelet = np.array(
                            datasetTrain.iloc[index].values)  # Ts contenente candidato shapelet
                        if (tree.warningDetected):
                            Dp = distanceProfile.naiveDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                                      window_size, TsToCompare)
                        else:
                            Dp = distanceProfile.massDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                                     window_size, TsToCompare)
                        minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
                        dfForDTreeTest[prefix + str(counter)].iloc[i] = minValueFromDProfile
                        if (booleanForTsAndStartingPos == True):
                            TsAndStartingPositionList.append([index, startingIndex])
                    counter += 1

        booleanForTsAndStartingPos = False #setto a false e smetto di raccogliere informazioni

    le = LabelEncoder()
    num_classes = le.fit_transform(dfForDTreeTest['class'])
    dfForDTreeTest['class'] = num_classes

    return dfForDTreeTest,TsAndStartingPositionList  # columnsList2 restituito per generare poi dFrame in "Split" (struttura dframe)



#dopo aver calcolato dfTrain, recupero la sottosequenza di ogni candidato shapelet
def retireveCandidatesSubSeq(tree,CandidatesList, dataset,window_size,numberOfMotifTrain, numberOfDiscordTrain):


    verboseretireveCandidatesSubSeq=False

    #genero colonne
    columnsList=list(['idTs','IdCandidate','startingPosition','M/D'])
    prefix='att'
    for i in range(window_size):
        columnsList.append(prefix + str(i))
    candDfMotifs=pd.DataFrame(columns=columnsList,index=range(numberOfMotifTrain))
    candDfDiscords = pd.DataFrame(columns=columnsList, index=range(numberOfDiscordTrain))

    CandidatesListM=CandidatesList[CandidatesList['M/D']==0]
    counter=0
    for j in range(len(CandidatesListM)):#CANDIDATI CON MOTIF
        #scandisco e inserisco prima tutte le subseq dei motif
        startingIndex = CandidatesListM.iloc[j]['startingPosition']  # indice di inizio del motif
        indexTsContainingCandidateShapelet=CandidatesListM.iloc[j]['IdTs']

        TsContainingCandidateShapelet=dataset[dataset['TsIndex']==indexTsContainingCandidateShapelet]
        TsContainingCandidateShapelet=TsContainingCandidateShapelet.values
        TsContainingCandidateShapelet=TsContainingCandidateShapelet[0][:-2]

        subSeqCandidate=TsContainingCandidateShapelet[startingIndex:startingIndex+window_size]

        candDfMotifs.iloc[counter]['idTs']=CandidatesListM.iloc[j]['IdTs']
        candDfMotifs.iloc[counter]['IdCandidate']=CandidatesListM.iloc[j]['IdCandidate']
        candDfMotifs.iloc[counter]['startingPosition'] = startingIndex
        candDfMotifs.iloc[counter]['M/D'] = 0
        for z in range(window_size):
            candDfMotifs.iloc[counter]['att'+str(z)] = subSeqCandidate[z]
        counter+=1

    CandidatesListD = CandidatesList[CandidatesList['M/D'] == 1]
    counter=0
    for j in range(len(CandidatesListD)):
        startingIndex = CandidatesListD.iloc[j]['startingPosition']   # indice di inizio del motif
        indexTsContainingCandidateShapelet = CandidatesListD.iloc[j]['IdTs']

        TsContainingCandidateShapelet = dataset[dataset['TsIndex'] == indexTsContainingCandidateShapelet]
        TsContainingCandidateShapelet = TsContainingCandidateShapelet.values
        TsContainingCandidateShapelet = TsContainingCandidateShapelet[0][:-2]

        subSeqCandidate = TsContainingCandidateShapelet[startingIndex:startingIndex + window_size]

        candDfDiscords.iloc[counter]['idTs'] = CandidatesListD.iloc[j]['IdTs']
        candDfDiscords.iloc[counter]['IdCandidate'] = CandidatesListD.iloc[j]['IdCandidate']
        candDfDiscords.iloc[counter]['startingPosition'] = startingIndex
        candDfDiscords.iloc[counter]['M/D'] = 1
        for z in range(window_size):
            candDfDiscords.iloc[counter]['att' + str(z)] = subSeqCandidate[z]
        counter+=1

    if(verboseretireveCandidatesSubSeq):
        print('subseq dei candidati estratti (rispettivamente motifs e poi discords)')
        #subseq dei candidati estratti (rispettivamente motifs e poi discords)
        print(candDfMotifs)
        print(candDfDiscords)

    EmptyDiscords = False
    CandidateMedoidsMotifs=[]
    CandidateMedoidsDiscords=[]

    if (len(candDfDiscords) == 0):
        EmptyDiscords=True


    # indici all interno di candDfMotifs & candDfDiscords dei candidati scelti come medoidi
    CandidateMedoidsMotifs = runKMeans(candDfMotifs, tree.n_clusters)
    if(verboseretireveCandidatesSubSeq):
        print('indici all interno di candDfMotifs & candDfDiscords dei candidati scelti come medoidi (rispettivamente motifs e poi discords) ')
        print(CandidateMedoidsMotifs)

    if(EmptyDiscords==False and numberOfDiscordTrain>tree.n_clusters):
        CandidateMedoidsDiscords = runKMeans(candDfDiscords,tree.n_clusters)
        if (verboseretireveCandidatesSubSeq):
            print(CandidateMedoidsDiscords)


    #riduco candDfMotifs & candDfDiscords mantenendo solo i candidati scelti
    candDfMotifs = candDfMotifs.iloc[CandidateMedoidsMotifs]
    candDfMotifs.reset_index(drop=True, inplace=True)
    if (EmptyDiscords == False and numberOfDiscordTrain>tree.n_clusters):
        candDfDiscords = candDfDiscords.iloc[CandidateMedoidsDiscords]
        candDfDiscords.reset_index(drop=True, inplace=True)

    if (verboseretireveCandidatesSubSeq):
        print('candDfMotifs & candDfDiscords mantenendo solo i candidati scelti')
        print(candDfMotifs)
    if (EmptyDiscords == False and numberOfDiscordTrain>tree.n_clusters):
        if (verboseretireveCandidatesSubSeq):
            print(candDfDiscords)

    #prendo l'id dei candidati da mantenere
    idTsCandidateMotifs = candDfMotifs['IdCandidate'].values
    if (EmptyDiscords == False and numberOfDiscordTrain>tree.n_clusters):
        idTsCandidateDiscords = candDfDiscords['IdCandidate'].values
    else:
        idTsCandidateDiscords=[]

    #determino i candidati da mantenere tra quelli estratti dal k-means, in base al CandidatesGroup scelto
    if(tree.candidatesGroup==2):
        ChosenCandidates = list(set(idTsCandidateMotifs) | set(idTsCandidateDiscords))
        # la loro unione potrebbe essere > tree.n_clusters (#medoidi scelti) quindi ne prendo solo tree.n_clusters
        ChosenCandidates = ChosenCandidates[:tree.n_clusters]
        # ora ho ridotto avendo CANDIDATI PROVENIENTI DA tree.n_clusters TS DIFFERENTI, MA POSSONO ESSERE ANCORA > tree.n_clusters

    elif(tree.candidatesGroup==0 or EmptyDiscords == True):
        ChosenCandidates=idTsCandidateMotifs #se ho scelto i motifs oppure non ci sono candidati discord estratti, mantengo i candidati medoidi scelti analizzando solo i candidati motifs
    else:
        ChosenCandidates = idTsCandidateDiscords #analogo per i discords

    if (verboseretireveCandidatesSubSeq):
        print('indice dei candidati da mantenere (scleti come medoidi) dentro CandidatesListTrain ')
        print(ChosenCandidates)

    #genero nuova lista candidati / finale
    CandidatesList = CandidatesList.iloc[ChosenCandidates]
    CandidatesList.reset_index(drop=True,inplace=True)

    return CandidatesList

#FUNZIONI PER PLOTTING DEI DATI



def plotData(Ts):
    Ts.plot(figsize=(7, 7), legend=None, title='Time series')
    plt.show()


def plot_motifs(mtfs, labels, ax, data, window_size):
    #data can be raw data or MP
    colori = 0
    colors = 'rmb'
    for ms,l in zip(mtfs,labels):
        c =colors[colori % len(colors)]
        starts = list(ms)
        ends = [min(s + window_size,len(data)-1) for s in starts]
        ax.plot(starts, data[starts],  c +'o',  label=l+'(Motif)')
        ax.plot(ends, data[ends],  c +'o', markerfacecolor='none')
        for nn in ms:
            ax.plot(range(nn,nn+window_size),data[nn:nn+window_size], c , linewidth=2)
        colori += 1

    #ax.plot(a,'green', linewidth=1, label="data") COMMENTATO PERCHE PLOTTO I DATI INDIPENDENTEMENTE
    ax.legend()


def plot_discords(dis, ax, data, window_size):
    # data can be raw data or Mp
    color = 'k'
    for start in dis:
        end = start + window_size
        ax.plot(start, data[start], color, label='Discord')
        if(end >= len(data)):
            end=len(data)-1
        ax.plot(end, data[end], color, markerfacecolor='none')

        ax.plot(range(start, start + window_size), data[start:start + window_size], color, linewidth=2)

    ax.legend(loc=1, prop={'size': 12})


def plot_all(Ts, mp, mot, motif_dist, dis, window_size):
    # genera e compara TS con MP, motifs e discords ottenuti

    # Append np.nan to Matrix profile to enable plotting against raw data (FILL DI 0 ALLA FINE PER RENDERE LE LUNGHEZZE UGUALI )
    mp_adj = np.append(mp, np.zeros(window_size - 1) + np.nan)

    # MODO 2 PER PLOTTARE (O-ORIENTED)
    # Plot dei dati
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 15))
    ax1.plot(np.arange(len(Ts)), Ts, label="Ts",
             color='g')  # stampo linespace su x e valori data su y (USATO SE NON STAMPO MOTIF/DIS)
    ax1.set_ylabel('Ts', size=22)
    plot_motifs(mot, [f"{md:.3f}" for md in motif_dist], ax1, Ts, window_size)  # sk
    plot_discords(dis, ax1, Ts, window_size)

    # Plot della Matrix Profile
    ax2.plot(np.arange(len(mp_adj)), mp_adj, label="Matrix Profile", color='green')
    ax2.set_ylabel('Matrix Profile', size=22)
    plot_motifs(mot, [f"{md:.3f}" for md in motif_dist], ax2, mp_adj, window_size)
    plot_discords(dis, ax2, mp_adj, window_size)

    plt.show()


def retrieve_all2(Ts,window_size,k):  # fornita la Ts calcola e restituisce mp, motifs, motifs_distances e discords
    Ts = Ts[:len(Ts)-1]  # rimuovo l'attributo "classe"

    dfMP = pd.DataFrame(Ts).astype(float)  # genero Dframe per lavorarci su, DA CAPIRE PERCHE SERVE FLOAT
    mp, mpi = matrixProfile.stomp(dfMP[0].values, window_size)  # OK STOMP

    # PREPARO TUPLA DA PASSARE ALLA FUN MOTIF (RICHIEDE TUPLA FATTA DA MP E MPI)
    tupla = mp, mpi

    mot, motif_dist = motifs.motifs(dfMP[0].values, tupla, k)

    # CALCOLO MOTIFS
    print('Motifs starting position: ' + str(mot) + ' Motifs values (min distances): ' + str(motif_dist))

    # CALCOLO DISCORDS
    dis = discords(mp, window_size, k)
    print('Discords starting position: ' + str(dis))

    tupla = mp, mot, motif_dist, dis
    return tupla


# riceve la lista di coppie dei motifs per ogni record(Ts), e resittuisce lista di valori singoli

def candidateFilter2(CandidateList):
    l2 = np.array([])
    for i in range(len(CandidateList['Motif'])):  # per ogni entry (per ogni record)
        numMotif = len(CandidateList['Motif'].iloc[i])
        # print(numMotif)
        for j in range(numMotif):  # per ogni lista di motif
            l1 = CandidateList['Motif'].iloc[i]  # prima lista
            l2 = np.append(l2, l1[j][0])  # prendo primo valore di ogni lista

        CandidateList['Motif'].iloc[i] = l2
        l2 = np.array([])  # svuoto array

    return CandidateList