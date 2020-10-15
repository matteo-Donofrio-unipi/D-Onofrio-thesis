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

class Tree:
    def __init__(self,candidatesGroup,maxDepth,minSamplesLeaf,removeUsedCandidate,window_size,k,n_clusters,warningDetected,verbose):
        self.candidatesGroup=candidatesGroup
        self.maxDepth=maxDepth
        self.minSamplesLeaf=minSamplesLeaf
        self.removeUsedCandidate=removeUsedCandidate
        self.window_size=window_size
        self.k=k
        self.n_clusters=n_clusters
        self.warningDetected=warningDetected
        self.verbose=verbose
        self.attributeList=list()

    def printTree(self):
        print(self.candidatesGroup)
        print(self.removeUsed)
        print('ciao')

    # dataset (dframe): nella riga i: indice della ts di appartenenza, distanza tra candidato e Ts, e classe di appartenenza di Ts
    # calcola entropia di un dataset basandosi sul num di classi esistenti
    def computeEntropy(self,dataset):
        value, counts = np.unique(dataset['class'], return_counts=True)
        actualEntropy = entropy(counts, base=2)
        return actualEntropy

    # calcola il gain tra entropia nodo padre e sommatoria entropia nodi figli (GAIN CALCOLATO SUL VALORE DELL'ATTRIBUTO)
    def computeGain(self,entropyParent, LenDatasetParent, Dleft, Dright):
        entropyLeft = self.computeEntropy(Dleft)
        entropyRight = self.computeEntropy(Dright)
        gain = entropyParent
        summation = (
                    ((len(Dleft) / LenDatasetParent) * entropyLeft) + ((len(Dright) / LenDatasetParent) * entropyRight))
        gain = gain - summation
        return gain




    # SPLIT SLAVE
    # effettua lo split del dataset sul attributo e valore fornito
    def split(self,dataset, attribute, value):
        columnsList = dataset.columns.values
        dizLeft = pd.DataFrame(columns=columnsList)
        dizRight = pd.DataFrame(columns=columnsList)
        attribute=str(attribute)
        value=str(value)

        q1=attribute+'<'+value
        q2=attribute+'>='+value

        dizLeft= dataset.query(q1)
        dizRight= dataset.query(q2)

        dizLeft = dizLeft.reset_index(drop=True)
        dizRight = dizRight.reset_index(drop=True)


        return dizLeft, dizRight




    # riceve dframe con mutual_information(gain) e in base al candidatesGroup scelto, determina il miglior attributo su cui splittare
    # che non è stato ancora utilizzato
    def getBestIndexAttribute(self,vecMutualInfo, CandidatesUsedListTrain, numberOfMotif, numberOfDiscord):
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
                print('gain: ' + str(vecMutualInfo.iloc[i]['gain']))
            else:
                i += 1

        return bestIndexAttribute, splitValue





    def computeMutualInfo(self,datasetForMutual, numberOfMotif, numberOfDiscord):
        # cerca attributo, il cui relativo best split value massimizza l'information gain nello split

        # definisco lista di indici inserire nella colonna 'attribute'
        if (self.candidatesGroup == 0):
            candidatesIndex = range(numberOfMotif)
            numAttributes = numberOfMotif
        elif (self.candidatesGroup == 1):
            candidatesIndex = range(numberOfMotif, numberOfMotif + numberOfDiscord)
            numAttributes = numberOfDiscord
        else:
            candidatesIndex = range(numberOfMotif + numberOfDiscord)
            numAttributes = numberOfMotif + numberOfDiscord

        columns = datasetForMutual.columns
        dframe = pd.DataFrame(columns=['attribute', 'splitValue', 'gain'],
                              index=range(len(columns) - 1))  # -1 cosi non prendo attr=class
        entropyParent = self.computeEntropy(datasetForMutual)

        # per ogni attributo, ordino il dframe sul suo valore
        # scandisco poi la y e appena cambia il valore di class effettuo uno split, memorizzando il best gain

        for i in range(len(columns) - 1):  # scandisco tutti gli attributi tranne 'class'
            bestGain = -1
            bestvalueForSplit = 0
            previousClass = -1  # deve essere settato ad un valore non presente nei class value
            attribute = columns[i]
            print('COMPUTE attr: ' + str(attribute))
            datasetForMutual = datasetForMutual.sort_values(by=attribute, ascending=True)

            y = datasetForMutual['class']

            for j in range(len(y)):
                if (j == 0):
                    previousClass = y[j]
                    continue
                else:
                    if (y[j] != previousClass):
                        testValue = datasetForMutual.iloc[j][attribute]
                        Dleft, Dright = self.split(datasetForMutual, attribute, testValue)
                        actualGain = self.computeGain(entropyParent, len(datasetForMutual), Dleft, Dright)
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
    def findBestAttributeValue(self,dataset, CandidatesUsedListTrain, numberOfMotif, numberOfDiscord,verbose):

        # cerca e restituisce attributo migliore su cui splittaree relativo valore ottimale (optimal split point)
        # CANDIDATE GROUP permette di scegliere se usare come candidati 0=motifs 1=discord 2=entrambi
        bestGain = 0
        actualGain = 0
        bestvalueForSplit = 0
        y = dataset['class'].values
        y = y.astype('int')
        entropyParent = self.computeEntropy(dataset)

        # trovo best Attribute
        numAttributes = len(dataset.columns.values)
        numAttributes -= 2  # tolgo i due attributi TsIndex e class dal Dframe
        datasetForMutual = pd.DataFrame()

        # preparo il Dframe da passare a mutual_info_classif, settando se scegliere tra motifs/discord/entrambi

        if (self.candidatesGroup == 0):  # solo motifs
            datasetForMutual = dataset.iloc[:, np.r_[:numberOfMotif]].copy()
        elif (self.candidatesGroup == 1):
            datasetForMutual = dataset.iloc[:, np.r_[numberOfMotif:numberOfMotif + numberOfDiscord]].copy()
        else:
            datasetForMutual = dataset.iloc[:, np.r_[:numAttributes]].copy()

        datasetForMutual['class'] = y

        # calcolo gain e miglior valore di split per ogni attributo

        vecMutualInfo = self.computeMutualInfo(datasetForMutual, numberOfMotif, numberOfDiscord)
        if (verbose == True):
            print('vec mutual info calcolato: ')
            print(vecMutualInfo)
        # se rimuovo candidati, faccio scegliere migliore non ancora utilizzato

        if (self.removeUsedCandidate == 1):
            indexBestAttribute, bestValueForSplit = self.getBestIndexAttribute(vecMutualInfo,
                                                                          CandidatesUsedListTrain, numberOfMotif,
                                                                          numberOfDiscord)
        else:  # se non rimuovo candidati, mi basta prendere il primo
            vecMutualInfo = vecMutualInfo.sort_values(by='gain', ascending=False)
            indexBestAttribute = vecMutualInfo.iloc[0]['attribute']
            bestValueForSplit = vecMutualInfo.iloc[0]['splitValue']
            print('gain: ' + str(vecMutualInfo.iloc[0]['gain']))  # stampo gain
        if (verbose == True):
            print('BEST attribute | value')
            print(indexBestAttribute, bestValueForSplit)

        splitValue = bestValueForSplit
        Dleft, Dright = self.split(dataset, 'cand'+str(indexBestAttribute), bestValueForSplit)

        return [indexBestAttribute, splitValue, Dleft, Dright]

    # SPLIT MASTER
    # funzione ricorsiva che implementa la creazione dell'albero di classificazione
    # memorizza in ogni nodo: attributo, valore attributo su cui splitto, entropia nodo, num pattern
    # memorizza in ogni foglia: entropia nodo, num pattern, classe nodo

    # VERSIONE CHE RIMUOVE I CANDIDATI QUANDO VENGONO SCELTI

    def buildTree(self,actualNode, dataset, depth, CandidatesUsedListTrain,
                  numberOfMotif, numberOfDiscord, verbose):
        # caso base: num pattern < soglia minima || profondità massima raggiunta => genero foglia con media delle classi
        # DATASET HA SEMPRE ALMENO UN PATTERN
        boolValue = self.checkIfIsLeaf(dataset)
        if (len(dataset) < self.minSamplesLeaf or depth >= self.maxDepth or boolValue == True):
            average = sum(dataset['class'].values) / len(dataset['class'].values)
            classValue = round(average)
            numPattern = len(dataset)
            entropy = self.computeEntropy(dataset)

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

            returnList = self.findBestAttributeValue(dataset, CandidatesUsedListTrain, numberOfMotif,
                                                numberOfDiscord, verbose)
            indexChosenAttribute = returnList[0]
            attributeValue = returnList[1]
            Dleft = returnList[2]
            Dright = returnList[3]
            numPattern = len(dataset)
            entropy = self.computeEntropy(dataset)
            self.attributeList.append(indexChosenAttribute)

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
                self.buildTree(actualNode.left, Dleft, depth + 1,
                          CandidatesUsedListTrain, numberOfMotif, numberOfDiscord, verbose)

            if (len(Dright) > 0):
                actualNode.right = Node(indexChosenAttribute)
                self.buildTree(actualNode.right, Dright, depth + 1,
                          CandidatesUsedListTrain, numberOfMotif, numberOfDiscord, verbose)




    # verifica se dataset, ha pattern appartenenti ad una sola classe => è gia foglia
    def checkIfIsLeaf(self,dataset):
        isLeaf = True
        entropy = self.computeEntropy(dataset)
        if (entropy > 0):
            isLeaf = False
        return isLeaf






    # effettua il primo passo dell'algo di generazione dell'albero, richiama ricorsivamente sui figli
    # VERSIONE CHE NON RIMUOVE I CANDIDATI QUANDO VENGONO SCELTI
    def fit(self,dfForDTree, CandidatesUsedListTrain, numberOfMotif,numberOfDiscord, verbose):
        # inizio algo per nodo radice
        returnList = self.findBestAttributeValue(dfForDTree, CandidatesUsedListTrain, numberOfMotif,
                                            numberOfDiscord, verbose)
        indexChosenAttribute = returnList[0]
        attributeValue = returnList[1]
        Dleft = returnList[2]
        Dright = returnList[3]
        self.attributeList.append(indexChosenAttribute)
        root = Node(indexChosenAttribute)
        numPattern = len(dfForDTree)
        entropy = self.computeEntropy(dfForDTree)

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
            self.buildTree(root.left, Dleft, 1, CandidatesUsedListTrain,
                      numberOfMotif, numberOfDiscord, verbose)
        if (len(Dright) > 0):
            self.buildTree(root.right, Dright, 1, CandidatesUsedListTrain,
                      numberOfMotif, numberOfDiscord, verbose)
        Tree.Root = root




    # stampa dell'albero
    def printAll(self,Root):
        if (Root.left == None and Root.right == None):
            print('foglia')
        print('Nodo: ' + str(Root.value))
        df = Root.data
        print(df)
        print("\n")
        if (Root.left != None):
            self.printAll(Root.left)
        if (Root.right != None):
            self.printAll(Root.right)

    def predict(self,testDataset, root):
        # preparo dataset
        numAttributes = len(testDataset.columns.values)
        numAttributes -= 2  # per prendere solo gli attributi utili a xTest
        yTest = testDataset.iloc[:]['class'].values
        yPredicted = np.zeros(len(yTest))
        xTest = testDataset.iloc[:, np.r_[:numAttributes]]

        # effettuo predizione per ogni pattern

        for i in range(len(xTest)):
            pattern = xTest.iloc[i]
            yPredicted[i] = self.treeExplorer(pattern, root)

        yTest = yTest.astype(int)
        yPredicted = yPredicted.astype(int)

        return yTest, yPredicted

    def treeExplorer(self,pattern, node):
        # caso base, node è foglia
        if (node.value == -1):
            return int(node.data[0])
        else:
            # caso ricorsivo
            attr = 'cand' + str(node.value)
            if (pattern[attr] < node.data[0]):
                return self.treeExplorer(pattern, node.left)
            else:
                return self.treeExplorer(pattern, node.right)
