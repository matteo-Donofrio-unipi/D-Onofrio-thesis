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


def setVariables(window_s,attributeL):
    global window_size
    window_size=window_s
    global attributeList
    attributeList=attributeL

def retrieve_all(Ts):  # fornita la Ts calcola e restituisce mp, motifs, motifs_distances e discords
    dfMP = pd.DataFrame(Ts).astype(float)  # genero Dframe per lavorarci su, DA CAPIRE PERCHE SERVE FLOAT
    mp, mpi = matrixProfile.stomp(dfMP[0].values, window_size)  # OK STOMP

    # PREPARO TUPLA DA PASSARE ALLA FUN MOTIF (RICHIEDE TUPLA FATTA DA MP E MPI)
    tupla = mp, mpi

    mot, motif_dist = motifs.motifs(dfMP[0].values, tupla, 2)

    # CALCOLO DISCORDS
    dis = discords(mp, window_size, 2)
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


def getDataStructures(df, verbose):
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
        mp, mot, motif_dist, dis = retrieve_all(Ts)
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
def computeSubSeqDistance(dataset, CandidatesList):
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



#dataset (dframe): nella riga i: indice della ts di appartenenza, distanza tra candidato e Ts, e classe di appartenenza di Ts
#calcola entropia di un dataset basandosi sul num di classi esistenti
def computeEntropy(dataset):
    value,counts = np.unique(dataset['class'], return_counts=True)
    actualEntropy=entropy(counts, base=2)
    return actualEntropy



#calcola il gain tra entropia nodo padre e sommatoria entropia nodi figli (GAIN CALCOLATO SUL VALORE DELL'ATTRIBUTO)
def computeGain(entropyParent,LenDatasetParent,Dleft,Dright):
    entropyLeft=computeEntropy(Dleft)
    entropyRight=computeEntropy(Dright)
    gain=entropyParent
    summation=( ((len(Dleft)/LenDatasetParent)*entropyLeft) +  ((len(Dright)/LenDatasetParent)*entropyRight) )
    gain=gain-summation
    return gain


#SPLIT SLAVE
#effettua lo split del dataset sul attributo e valore fornito
def split(dataset,attribute,value):
    columnsList=dataset.columns.values
    dizLeft=pd.DataFrame(columns=columnsList)
    dizRight=pd.DataFrame(columns=columnsList)
    for i in range(len(dataset)):
        if dataset.iloc[i][attribute] < value:
            dizLeft = dizLeft.append(dataset.iloc[i], ignore_index=True)
        else:
            dizRight = dizRight.append(dataset.iloc[i], ignore_index=True)
    return dizLeft, dizRight


# riceve dframe con mutual_information(gain) e in base al candidatesGroup scelto, determina il miglior attributo su cui splittare
# che non è stato ancora utilizzato
def getBestIndexAttribute(vecMutualInfo, candidatesGroup, CandidatesUsedListTrain, numberOfMotif, numberOfDiscord):
    # ordino i candidati in base a gain decrescente

    vecMutualInfo = vecMutualInfo.sort_values(by='gain', ascending=False)

    # scandisco i candidati fino a trovare il candidato con miglior gain che non è ancora stato usato

    bestIndexAttribute = -1
    i = 0

    # cicla fin quando trova candidato libero con gain maggiore
    while (bestIndexAttribute == -1 and i < len(vecMutualInfo)):
        attributeToVerify = int(vecMutualInfo.iloc[i]['attribute'])
        if (CandidatesUsedListTrain.iloc[attributeToVerify]['Used'] == False):
            bestIndexAttribute = attributeToVerify
            splitValue = vecMutualInfo.iloc[i]['splitValue']
            CandidatesUsedListTrain.iloc[
                attributeToVerify] = True  # settando a true il candidato scelto, non sarà usato in seguito
            print('gain: '+str(vecMutualInfo.iloc[i]['gain']))
        else:
            i += 1

    return bestIndexAttribute, splitValue


def computeMutualInfo(datasetForMutual, candidatesGroup, numberOfMotif, numberOfDiscord):
    # cerca attributo, il cui relativo best split value massimizza l'information gain nello split

    # definisco lista di indici inserire nella colonna 'attribute'
    if (candidatesGroup == 0):
        candidatesIndex = range(numberOfMotif)
        numAttributes = numberOfMotif
    elif (candidatesGroup == 1):
        candidatesIndex = range(numberOfMotif, numberOfMotif + numberOfDiscord)
        numAttributes = numberOfDiscord
    else:
        candidatesIndex = range(numberOfMotif + numberOfDiscord)
        numAttributes = numberOfMotif + numberOfDiscord

    columns = datasetForMutual.columns
    dframe = pd.DataFrame(columns=['attribute', 'splitValue', 'gain'],
                          index=range(len(columns) - 1))  # -1 cosi non prendo attr=class
    entropyParent = computeEntropy(datasetForMutual)

    # per ogni attributo, ordino il dframe sul suo valore
    # scandisco poi la y e appena cambia il valore di class effettuo uno split, memorizzando il best gain

    for i in range(len(columns) - 1):  # scandisco tutti gli attributi tranne 'class'
        bestGain = -1
        bestvalueForSplit = 0
        previousClass = -1  # deve essere settato ad un valore non presente nei class value
        attribute = columns[i]
        print('COMPUTE attr: '+str(attribute))
        datasetForMutual = datasetForMutual.sort_values(by=attribute, ascending=True)

        y = datasetForMutual['class']

        for j in range(len(y)):
            if (j == 0):
                previousClass = y[j]
                continue
            else:
                if (y[j] != previousClass):
                    testValue = datasetForMutual.iloc[j][attribute]
                    Dleft, Dright = split(datasetForMutual, attribute, testValue)
                    actualGain = computeGain(entropyParent, len(datasetForMutual), Dleft, Dright)
                    if (actualGain > bestGain):
                        bestGain = actualGain
                        bestvalueForSplit = testValue

                previousClass = y[j]
                # memorizzo in posizione i-esima lo split migliore e relativo gain

        dframe.iloc[i]['splitValue'] = bestvalueForSplit
        dframe.iloc[i]['gain'] = bestGain

    dframe['attribute'] = candidatesIndex

    return dframe


# SPLIT INTERMEDIO
# dato il dataset, cerca il miglior attributo e relativo valore (optimal split point) su cui splittare
# restituiendo il dataset splittato e i valori trovati
def findBestAttributeValue(dataset, candidatesGroup, CandidatesUsedListTrain, numberOfMotif, numberOfDiscord,
                           removeUsedCandidate, verbose):

    # cerca e restituisce attributo migliore su cui splittaree relativo valore ottimale (optimal split point)
    # CANDIDATE GROUP permette di scegliere se usare come candidati 0=motifs 1=discord 2=entrambi
    bestGain = 0
    actualGain = 0
    bestvalueForSplit = 0
    y = dataset['class'].values
    y = y.astype('int')
    entropyParent = computeEntropy(dataset)

    # trovo best Attribute
    numAttributes = len(dataset.columns.values)
    numAttributes -= 2  # tolgo i due attributi TsIndex e class dal Dframe
    datasetForMutual = pd.DataFrame()

    # preparo il Dframe da passare a mutual_info_classif, settando se scegliere tra motifs/discord/entrambi

    if (candidatesGroup == 0):  # solo motifs
        datasetForMutual = dataset.iloc[:, np.r_[:numberOfMotif]].copy()
    elif (candidatesGroup == 1):
        datasetForMutual = dataset.iloc[:, np.r_[numberOfMotif:numberOfMotif + numberOfDiscord]].copy()
    else:
        datasetForMutual = dataset.iloc[:, np.r_[:numAttributes]].copy()

    datasetForMutual['class'] = y

    # calcolo gain e miglior valore di split per ogni attributo

    vecMutualInfo = computeMutualInfo(datasetForMutual, candidatesGroup, numberOfMotif, numberOfDiscord)
    if (verbose == True):
        print('vec mutual info calcolato: ')
        print(vecMutualInfo)
    # se rimuovo candidati, faccio scegliere migliore non ancora utilizzato

    if (removeUsedCandidate == 1):
        indexBestAttribute, bestValueForSplit = getBestIndexAttribute(vecMutualInfo, candidatesGroup,
                                                                      CandidatesUsedListTrain, numberOfMotif,
                                                                      numberOfDiscord)
    else:  # se non rimuovo candidati, mi basta prendere il primo
        vecMutualInfo = vecMutualInfo.sort_values(by='gain', ascending=False)
        indexBestAttribute = vecMutualInfo.iloc[0]['attribute']
        bestValueForSplit = vecMutualInfo.iloc[0]['splitValue']
        print('gain: '+str(vecMutualInfo.iloc[0]['gain'])) #stampo gain
    if (verbose == True):
        print('BEST attribute | value')
        print(indexBestAttribute, bestValueForSplit)

    splitValue = bestValueForSplit
    Dleft, Dright = split(dataset, indexBestAttribute, bestValueForSplit)

    return [indexBestAttribute, splitValue, Dleft, Dright]


# SPLIT MASTER
# funzione ricorsiva che implementa la creazione dell'albero di classificazione
# memorizza in ogni nodo: attributo, valore attributo su cui splitto, entropia nodo, num pattern
# memorizza in ogni foglia: entropia nodo, num pattern, classe nodo

# VERSIONE CHE RIMUOVE I CANDIDATI QUANDO VENGONO SCELTI

def buildTree(actualNode, dataset, maxDepth, minSamplesLeaf, depth, candidatesGroup, CandidatesUsedListTrain,
              numberOfMotif, numberOfDiscord, removeUsedCandidate, verbose):
    # caso base: num pattern < soglia minima || profondità massima raggiunta => genero foglia con media delle classi
    # DATASET HA SEMPRE ALMENO UN PATTERN
    boolValue = checkIfIsLeaf(dataset)
    if (len(dataset) < minSamplesLeaf or depth >= maxDepth or boolValue == True):
        average = sum(dataset['class'].values) / len(dataset['class'].values)
        classValue = round(average)
        numPattern = len(dataset)
        entropy = computeEntropy(dataset)

        nodeInfo = list()
        nodeInfo.append(classValue)
        nodeInfo.append(numPattern)
        nodeInfo.append(entropy)

        actualNode.data = nodeInfo
        actualNode.value = -1
        actualNode.left = None
        actualNode.right = None
        return
        # caso ricorsivo in cui si può splittare
    else:

        returnList = findBestAttributeValue(dataset, candidatesGroup, CandidatesUsedListTrain, numberOfMotif,
                                            numberOfDiscord, removeUsedCandidate, verbose)
        indexChosenAttribute = returnList[0]
        attributeValue = returnList[1]
        Dleft = returnList[2]
        Dright = returnList[3]
        numPattern = len(dataset)
        entropy = computeEntropy(dataset)
        attributeList.append(indexChosenAttribute)

        # memorizzo nel nodo l'attributo, il valore e altre info ottenute dallo split

        nodeInfo = list()
        nodeInfo.append(attributeValue)
        nodeInfo.append(numPattern)
        nodeInfo.append(entropy)
        actualNode.data = nodeInfo
        actualNode.value = (indexChosenAttribute)

        # se possibile richiamo ricorsivamente sul nodo dx e sx figlio
        if (len(Dleft) > 0):
            actualNode.left = Node(indexChosenAttribute)
            buildTree(actualNode.left, Dleft, maxDepth, minSamplesLeaf, depth + 1, candidatesGroup,
                      CandidatesUsedListTrain, numberOfMotif, numberOfDiscord, removeUsedCandidate, verbose)

        if (len(Dright) > 0):
            actualNode.right = Node(indexChosenAttribute)
            buildTree(actualNode.right, Dright, maxDepth, minSamplesLeaf, depth + 1, candidatesGroup,
                      CandidatesUsedListTrain, numberOfMotif, numberOfDiscord, removeUsedCandidate, verbose)


#verifica se dataset, ha pattern appartenenti ad una sola classe => è gia foglia
def checkIfIsLeaf(dataset):
    isLeaf=True
    entropy=computeEntropy(dataset)
    if(entropy>0):
        isLeaf=False
    return isLeaf


# effettua il primo passo dell'algo di generazione dell'albero, richiama ricorsivamente sui figli
# VERSIONE CHE NON RIMUOVE I CANDIDATI QUANDO VENGONO SCELTI
def fit(dfForDTree, candidatesGroup, CandidatesUsedListTrain, maxDepth, minSamplesLeaf, numberOfMotif,
              numberOfDiscord, removeUsedCandidate, verbose):
    # inizio algo per nodo radice
    returnList = findBestAttributeValue(dfForDTree, candidatesGroup, CandidatesUsedListTrain, numberOfMotif,
                                        numberOfDiscord, removeUsedCandidate, verbose)
    indexChosenAttribute = returnList[0]
    attributeValue = returnList[1]
    Dleft = returnList[2]
    Dright = returnList[3]
    attributeList.append(indexChosenAttribute)
    root = Node(indexChosenAttribute)
    numPattern = len(dfForDTree)
    entropy = computeEntropy(dfForDTree)

    # memorizzo nel nodo l'attributo, il valore e altre info ottenute dallo split

    nodeInfo = list()
    nodeInfo.append(attributeValue)
    nodeInfo.append(numPattern)
    nodeInfo.append(entropy)
    root.data = nodeInfo

    root.left = Node(indexChosenAttribute)
    root.right = Node(indexChosenAttribute)

    # chiamata ricorsiva
    if (len(Dleft) > 0):
        buildTree(root.left, Dleft, maxDepth, minSamplesLeaf, 1, candidatesGroup, CandidatesUsedListTrain,
                  numberOfMotif, numberOfDiscord, removeUsedCandidate, verbose)
    if (len(Dright) > 0):
        buildTree(root.right, Dright, maxDepth, minSamplesLeaf, 1, candidatesGroup, CandidatesUsedListTrain,
                  numberOfMotif, numberOfDiscord, removeUsedCandidate, verbose)
    return root





#stampa dell'albero
def printAll(Root):
    if(Root.left==None and Root.right==None):
        print('foglia')
    print('Nodo: '+str(Root.value))
    df=Root.data
    print(df)
    print("\n")
    if(Root.left!=None):
        printAll(Root.left)
    if(Root.right!=None):
        printAll(Root.right)


def predict(testDataset, root):
    # preparo dataset
    numAttributes = len(testDataset.columns.values)
    numAttributes -= 2  # per prendere solo gli attributi utili a xTest
    yTest = testDataset.iloc[:]['class'].values
    yPredicted = np.zeros(len(yTest))
    xTest = testDataset.iloc[:, np.r_[:numAttributes]]

    # effettuo predizione per ogni pattern

    for i in range(len(xTest)):
        pattern = xTest.iloc[i]
        yPredicted[i] = treeExplorer(pattern, root)

    yTest = yTest.astype(int)
    yPredicted = yPredicted.astype(int)

    return yTest, yPredicted





def treeExplorer(pattern,node):
    #caso base, node è foglia
    if(node.value==-1):
        return int(node.data[0])
    else:
    #caso ricorsivo
        attr='cand'+str(node.value)
        if(pattern[attr] < node.data[0]):
            return treeExplorer(pattern,node.left)
        else:
            return treeExplorer(pattern,node.right)


def computeSubSeqDistanceForTest(datasetTest, datasetTrain, attributeList, CandidatesList, numberOfMotif,
                                 numberOfDiscord):
    # quantifico il num di candidati usati dall'albero e in base a tale valore genero colonne per dfForDTree
    # quantifico il num di candidati e in base a tale valore genero colonne per dfForDTree
    columnsList2 = list()
    prefix = 'cand'
    TsAndStartingPositionList=list() #contiene le coppie (Ts, startingPosition) per tenere traccia degli shaplet ottenuti
    booleanForTsAndStartingPos=True #dopo aver raccolto dati nel primo giro, lo setto a false e mi fermo (altrimenti raccolgo stessi dati)
    for i in attributeList:
        columnsList2.append(prefix + str(i))
    columnsList2.append('TsIndex')
    columnsList2.append('class')
    dfForDTree = pd.DataFrame(columns=columnsList2, index=range(0, len(datasetTest)))

    # per ogni Ts, scandisco ogni candidato e calcolo la distanza minore
    for i in range(len(datasetTest)):
        # acquisisco la Ts
        TsToCompare = np.array(datasetTest.iloc[i].values)
        classValue = TsToCompare[len(TsToCompare) - 1]  # la classe è sempre il penultimo attributo
        TsToCompare = TsToCompare[:len(TsToCompare) - 2]  # la serie è ottenuta rimuovendo i due ultimi attributi
        #I VALORI (-1, -2) SONO DIVERSI DA QUELLI USATI IN COMPUTE NORMALE, PERCHE QUI NON PASSO LA STRUTTURA A GETDATASTRUCTURES => NON AGGIUNGO COLONNA TS INDEX
        dfForDTree['TsIndex'].iloc[i] = i
        dfForDTree['class'].iloc[i] = classValue
        counter = 0
        # scandisco e calcolo distanza dai motif
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
                        dfForDTree[prefix + str(counter)].iloc[i] = minValueFromDProfile
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
                        dfForDTree[prefix + str(counter)].iloc[i] = minValueFromDProfile
                        if (booleanForTsAndStartingPos == True):
                            TsAndStartingPositionList.append([j, startingIndex])
                    counter += 1

        booleanForTsAndStartingPos = False #setto a false e smetto di raccogliere informazioni

    le = LabelEncoder()
    num_classes = le.fit_transform(dfForDTree['class'])
    dfForDTree['class'] = num_classes

    return dfForDTree,TsAndStartingPositionList  # columnsList2 restituito per generare poi dFrame in "Split" (struttura dframe)



