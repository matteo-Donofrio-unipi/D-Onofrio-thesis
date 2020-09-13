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


# LIBRERIA PER TUTTE LE FUNZIONI DI PRE PROCESSING DEI DATI
# DALL'ESTRAZIONE AL PASSAGGIO PER LA CREAZIONE DELL'ALBERO



def computeLoadedDataset(X, y):

    columnsList = np.arange(len(X[0]))
    columnsList2 = list()
    lastAttribute = ['TsIndex', 'class']
    prefix = 'att'

    for i in columnsList:
        columnsList2.append(prefix + str(i))
    columnsList2.append('target')
    dataset = pd.DataFrame(columns=columnsList2, index=range(0, len(X)))

    for i in range(len(X)):
        l1 = list()
        record = X[i]
        for j in range(len(X[i])):
            l1.append(record[j][0])
        l1.append(y[i])
        dataset.iloc[i] = l1

    return dataset

def retrieve_all(Ts,window_size,k):  # fornita la Ts calcola e restituisce mp, motifs, motifs_distances e discords
    dfMP = pd.DataFrame(Ts).astype(float)  # genero Dframe per lavorarci su, DA CAPIRE PERCHE SERVE FLOAT
    mp, mpi = matrixProfile.stomp(dfMP[0].values, window_size)  # OK STOMP

    # PREPARO TUPLA DA PASSARE ALLA FUN MOTIF (RICHIEDE TUPLA FATTA DA MP E MPI)
    tupla = mp, mpi

    mot, motif_dist = motifs.motifs(dfMP[0].values, tupla, k)

    # CALCOLO DISCORDS
    dis = discords(mp, window_size, k)
    # print('Discords starting position: '+str(dis))

    tupla = mp, mot, motif_dist, dis
    return tupla



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



# riceve la lista di candidati e genera una lista della stessa dimensione di booleani
def buildCandidatesUsedList(CandidatesList, numberOfMotif, numberOfDiscord):
    CandidatesUsedList = pd.DataFrame(columns=['Used'], index=range(0, numberOfMotif + numberOfDiscord))
    boolList = [False] * (numberOfMotif + numberOfDiscord)
    CandidatesUsedList['Used'] = boolList
    return CandidatesUsedList



def getDataStructures(df,window_size,k,verbose):
    # trasformo da stringa a numero il campo target
    le = LabelEncoder()
    num_classes = le.fit_transform(df['target'])
    df['target'] = num_classes
    df['TsIndex'] = np.arange(len(df))
    # diz={'Motif':[],'Motif-Dist':[],'Discord':[]}
    diz = {'Motif': [], 'Discord': []}

    # CALCOLO MOTIF E DISCORD E LI INSERISCO NEL DIZIONARIO
    for i in range(len(df)):
        Ts = np.array(df.iloc[i][:-2].values) #-2 perche rimuovo l'attributo target e index inserito precedentemente
        mp, mot, motif_dist, dis = retrieve_all(Ts,window_size,k)
        diz['Motif'].insert(i, mot)
        diz['Discord'].insert(i, dis)

    # GENERO DFRAME DA DIZIONARIO

    CandidatesList = pd.DataFrame(diz)
    CandidatesList, numberOfMotif, numberOfDiscord = candidateFilter(CandidatesList)
    CandidatesUsedList = buildCandidatesUsedList(CandidatesList, numberOfMotif, numberOfDiscord)

    if (verbose == True):
        print('Candidati estratti')
        print(CandidatesList)
        print(numberOfMotif, numberOfDiscord)
        print(CandidatesUsedList)

    return mp, CandidatesList, numberOfMotif, numberOfDiscord, CandidatesUsedList





# per ogni Ts calcolo Dprofile con ogni candidato e inserisco la distanza minima con candidato i-esimo nella colonna i-esima
def computeSubSeqDistance(dataset, CandidatesList,window_size):
    # quantifico il num di candidati e in base a tale valore genero colonne per dfForDTree
    numberOfCandidates = 0
    for i in range(len(CandidatesList)):
        numberOfCandidates += len(CandidatesList['Motif'].loc[i])
        numberOfCandidates += len(CandidatesList['Discord'].loc[i])
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
        for j in range(len(CandidatesList)):
            numMotif = len(CandidatesList['Motif'].iloc[j])
            # scandisco e calcolo distanza dai motif
            for k in range(numMotif):
                l1 = CandidatesList['Motif'].iloc[j]  # lista di indice i in motifDiscordList
                startingIndex = l1[k]  # indice di inizio del motif
                TsContainingCandidateShapelet = np.array(dataset.iloc[j].values)  # Ts contenente candidato shapelet
                Dp = distanceProfile.massDistanceProfile(TsContainingCandidateShapelet, int(startingIndex), window_size,TsToCompare)
                minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
                dfForDTree[prefix + str(counter)].iloc[i] = minValueFromDProfile
                counter += 1
        for j in range(len(CandidatesList)):
            numDiscord = len(CandidatesList['Discord'].iloc[j])
            # scandisco e calcolo distanza dai discord
            for k in range(numDiscord):
                l1 = CandidatesList['Discord'].iloc[j]  # lista di indice i in motifDiscordList
                startingIndex = l1[k]  # indice di inizio del motif
                TsContainingCandidateShapelet = np.array(dataset.iloc[j].values)  # Ts contenente candidato shapelet
                Dp = distanceProfile.massDistanceProfile(TsContainingCandidateShapelet, int(startingIndex), window_size,
                                                         TsToCompare)
                minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
                dfForDTree[prefix + str(counter)].iloc[i] = minValueFromDProfile
                counter += 1

    # print(counter)
    return dfForDTree  # columnsList2 restituito per generare poi dFrame in "Split" (struttura dframe)




def computeSubSeqDistanceForTest(datasetTest, datasetTrain, attributeList, CandidatesList, numberOfMotif,
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
            for j in range(len(CandidatesList)):
                numMotif = len(CandidatesList['Motif'].iloc[j])
                for k in range(numMotif):
                    if (counter == candidateIndex):
                        l1 = CandidatesList['Motif'].iloc[j]  # lista di indice i in motifDiscordList
                        startingIndex = l1[k]  # indice di inizio del motif
                        TsContainingCandidateShapelet = np.array(
                            datasetTrain.iloc[j].values)  # Ts contenente candidato shapelet
                        Dp = distanceProfile.massDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                                 window_size, TsToCompare)
                        minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
                        dfForDTreeTest[prefix + str(counter)].iloc[i] = minValueFromDProfile
                        if(booleanForTsAndStartingPos==True):
                            TsAndStartingPositionList.append([j,startingIndex])
                    counter += 1
            for j in range(len(CandidatesList)):
                numDiscord = len(CandidatesList['Discord'].iloc[j])
                for k in range(numDiscord):
                    if (counter == candidateIndex):
                        l1 = CandidatesList['Discord'].iloc[j]  # lista di indice i in motifDiscordList
                        startingIndex = l1[k]  # indice di inizio del motif
                        TsContainingCandidateShapelet = np.array(
                            datasetTrain.iloc[j].values)  # Ts contenente candidato shapelet
                        Dp = distanceProfile.massDistanceProfile(TsContainingCandidateShapelet, int(startingIndex),
                                                                 window_size, TsToCompare)
                        minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
                        dfForDTreeTest[prefix + str(counter)].iloc[i] = minValueFromDProfile
                        if (booleanForTsAndStartingPos == True):
                            TsAndStartingPositionList.append([j, startingIndex])
                    counter += 1

        booleanForTsAndStartingPos = False #setto a false e smetto di raccogliere informazioni

    le = LabelEncoder()
    num_classes = le.fit_transform(dfForDTreeTest['class'])
    dfForDTreeTest['class'] = num_classes

    return dfForDTreeTest,TsAndStartingPositionList  # columnsList2 restituito per generare poi dFrame in "Split" (struttura dframe)


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