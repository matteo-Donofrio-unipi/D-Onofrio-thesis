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
from PreProcessingLibrary2 import  computeSubSeqDistance,reduceNumberCandidates,plotDataAndShapelet


class Tree:
    def __init__(self,candidatesGroup,maxDepth,minSamplesLeaf,removeUsedCandidate,window_size,k,useClustering,n_clusters,warningDetected, verbose ):
        self.candidatesGroup=candidatesGroup
        self.maxDepth=maxDepth
        self.minSamplesLeaf=minSamplesLeaf
        self.removeUsedCandidate=removeUsedCandidate
        self.window_size=window_size
        self.k=k
        self.useClustering=useClustering
        self.n_clusters=n_clusters
        self.OriginalCandidatesUsedListTrain=[]
        self.warningDetected=warningDetected
        self.verbose=verbose
        self.attributeList=list()


    def modifyCandidateUsedList(self,newCand):
        self.OriginalCandidatesUsedListTrain=newCand

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
        attribute=str(attribute)
        #value=str(value)

        dizLeft=dataset[dataset[int(attribute)] < value]
        dizRight = dataset[dataset[int(attribute)] >= value]

        dizLeft = dizLeft.reset_index(drop=True)
        dizRight = dizRight.reset_index(drop=True)


        return dizLeft, dizRight




    # riceve dframe con mutual_information(gain) e in base al candidatesGroup scelto, determina il miglior attributo su cui splittare
    # che non è stato ancora utilizzato
    def getBestIndexAttribute(self,CandidatesUsedListTrain,vecMutualInfo,verbose):
        # ordino i candidati in base a gain decrescente

        vecMutualInfo = vecMutualInfo.sort_values(by='gain', ascending=False)

        # scandisco i candidati fino a trovare il candidato con miglior gain che non è ancora stato usato

        bestIndexAttribute = -1
        i = 0

        # cicla fin quando trova candidato libero con gain maggiore
        while (bestIndexAttribute == -1 and i < len(vecMutualInfo)):
            attributeToVerify = int(vecMutualInfo.iloc[i]['IdCandidate'])
            if (CandidatesUsedListTrain.loc[attributeToVerify]['Used']==False):
                bestIndexAttribute = attributeToVerify
                splitValue = vecMutualInfo.iloc[i]['splitValue']

                CandidatesUsedListTrain.loc[CandidatesUsedListTrain["IdCandidate"]==attributeToVerify,"Used"] = True  # settando a true il candidato scelto, non sarà usato in seguito
                if (verbose):
                    print('gain: ' + str(vecMutualInfo.iloc[i]['gain']))
            else:
                i += 1

        return bestIndexAttribute, splitValue





    def computeMutualInfo(self,datasetForMutual,verbose):
        # cerca attributo, il cui relativo best split value massimizza l'information gain nello split


        columns = datasetForMutual.columns
        dframe = pd.DataFrame(columns=['IdCandidate', 'splitValue', 'gain'],
                              index=range(len(columns) - 2))  # -1 cosi non prendo attr=class e TsIndex
        entropyParent = self.computeEntropy(datasetForMutual)

        # per ogni attributo, ordino il dframe sul suo valore
        # scandisco poi la y e appena cambia il valore di class effettuo uno split, memorizzando il best gain

        for i in range(len(columns) - 2):  # scandisco tutti gli attributi tranne 'class'
            bestGain = -1
            bestvalueForSplit = 0
            previousClass = -1  # deve essere settato ad un valore non presente nei class value
            attribute = columns[i]
            if (verbose):
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

        dframe['IdCandidate'] = columns[:-2]

        return dframe

    # SPLIT INTERMEDIO
    # dato il dataset, cerca il miglior attributo e relativo valore (optimal split point) su cui splittare
    # restituiendo il dataset splittato e i valori trovati
    def findBestSplit(self,dfForDTree, verbose=False):

        # cerca e restituisce attributo migliore su cui splittaree relativo valore ottimale (optimal split point)
        # CANDIDATE GROUP permette di scegliere se usare come candidati 0=motifs 1=discord 2=entrambi

        # trovo best Attribute

        # calcolo gain e miglior valore di split per ogni attributo

        vecMutualInfo = self.computeMutualInfo(dfForDTree,verbose)
        if (verbose == True):
            print('vec mutual info calcolato: ')
            print(vecMutualInfo)

        # se rimuovo candidati, faccio scegliere migliore non ancora utilizzato
        if (self.removeUsedCandidate == 1):
            indexBestAttribute, bestValueForSplit  = self.getBestIndexAttribute(self.OriginalCandidatesUsedListTrain,vecMutualInfo,verbose)
        else:  # se non rimuovo candidati, mi basta prendere il primo
            vecMutualInfo = vecMutualInfo.sort_values(by='gain', ascending=False)
            indexBestAttribute = vecMutualInfo.iloc[0]['IdCandidate']
            bestValueForSplit = vecMutualInfo.iloc[0]['splitValue']
            if (verbose):
                print('gain: ' + str(vecMutualInfo.iloc[0]['gain']))  # stampo gain

        if (verbose == True):
            print('BEST attribute | value')
            print(indexBestAttribute, bestValueForSplit)

        splitValue = bestValueForSplit
        Dleft, Dright = self.split(dfForDTree,str(indexBestAttribute), bestValueForSplit)

        return [indexBestAttribute, splitValue, Dleft, Dright]

    # SPLIT MASTER
    # funzione ricorsiva che implementa la creazione dell'albero di classificazione
    # memorizza in ogni nodo: attributo, valore attributo su cui splitto, entropia nodo, num pattern
    # memorizza in ogni foglia: entropia nodo, num pattern, classe nodo

    # VERSIONE CHE RIMUOVE I CANDIDATI QUANDO VENGONO SCELTI

    def buildTree(self,actualNode, dataset, depth,verbose):
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

            returnList = self.findBestSplit(dataset,verbose)
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
            actualNode.value = (int(indexChosenAttribute))

            if (verbose):
                print('DLEFT & DRIGHT INIZIALMENTE')
                print(Dleft)
                print(Dright)

            if(self.useClustering):

                # effettuo clustering
                TsIndexLeft = Dleft['TsIndex']  # TsIndex contenute in Dleft

                CandidatesListLeft = self.OriginalCandidatesListTrain['IdTs'].isin(
                    TsIndexLeft)  # setta a True gli indici dei candidati che sono stati generati dalle Ts contenute in Dleft

                CandidateToCluster = self.OriginalCandidatesListTrain[
                    CandidatesListLeft]  # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dleft
                CandidateToCluster = CandidateToCluster.reset_index(drop=True)

                indexChoosenMedoids = reduceNumberCandidates(self, CandidateToCluster,
                                                             returnOnlyIndex=True)  # indici di OriginalCandidatesListTrain conteneti candidati da mantenere

                CandidateToCluster = CandidateToCluster.iloc[indexChoosenMedoids] #candidati da mantenere

                if (verbose):
                    print('CANDIDATI RIMASTI IN BUILD')
                    print(CandidateToCluster)

                # calcolo distanze tra Ts in Dleft e candidati scelti
                Dleft = computeSubSeqDistance(self, TsIndexLeft, CandidateToCluster, self.window_size)

                # RIPETO PER DRIGHT------------------------------------------------------

                TsIndexRight = Dright['TsIndex']  # TsIndex contenute in Dleft

                CandidatesListRight = self.OriginalCandidatesListTrain['IdTs'].isin(
                    TsIndexRight)  # # setta a True gli indici dei candidati che sono stati generati dalle Ts contenute in Dright

                CandidateToCluster = self.OriginalCandidatesListTrain[
                    CandidatesListRight]   # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dright
                CandidateToCluster = CandidateToCluster.reset_index(drop=True)

                indexChoosenMedoids = reduceNumberCandidates(self, CandidateToCluster,
                                                             returnOnlyIndex=True)  # indici di OriginalCandidatesListTrain conteneti candidati da mantenere

                CandidateToCluster = CandidateToCluster.iloc[indexChoosenMedoids] #candidati da mantenere

                if (verbose):
                    print('CANDIDATI RIMASTI IN BUILD')
                    print(CandidateToCluster)

                #calcolo distanze tra Ts in Dright e candidati scelti
                Dright = computeSubSeqDistance(self, TsIndexRight, CandidateToCluster, self.window_size)

                if (verbose):
                    print('DLEFT & DRIGHT DOPO IL CLUSTERING')
                    print(Dleft)
                    print(Dright)



            # se possibile richiamo ricorsivamente sul nodo dx e sx figlio
            if (len(Dleft) > 0):
                actualNode.left = Node(int(indexChosenAttribute))
                self.buildTree(actualNode.left, Dleft, depth + 1, verbose)

            if (len(Dright) > 0):
                actualNode.right = Node(int(indexChosenAttribute))
                self.buildTree(actualNode.right, Dright, depth + 1, verbose)




    # verifica se dataset, ha pattern appartenenti ad una sola classe => è gia foglia
    def checkIfIsLeaf(self,dataset):
        isLeaf = True
        entropy = self.computeEntropy(dataset)
        if (entropy > 0):
            isLeaf = False
        return isLeaf






    # effettua il primo passo dell'algo di generazione dell'albero, richiama ricorsivamente sui figli
    # VERSIONE CHE NON RIMUOVE I CANDIDATI QUANDO VENGONO SCELTI
    def fit(self,dfForDTree,verbose):
        # inizio algo per nodo radice
        returnList = self.findBestSplit(dfForDTree,False)
        indexChosenAttribute = returnList[0]
        attributeValue = returnList[1]
        Dleft = returnList[2]
        Dright = returnList[3]
        self.attributeList.append(int(indexChosenAttribute))
        root = Node(int(indexChosenAttribute))
        numPattern = len(dfForDTree)
        entropy = self.computeEntropy(dfForDTree)

        # memorizzo nel nodo l'attributo, il valore e altre info ottenute dallo split

        nodeInfo = list()
        nodeInfo.append(attributeValue)
        nodeInfo.append(numPattern)
        nodeInfo.append(entropy)
        root.data = nodeInfo

        root.left = Node(int(indexChosenAttribute))
        root.right = Node(int(indexChosenAttribute))

        if (verbose):
            print('DLEFT & DRIGHT INIZIALMENTE')
            print(Dleft)
            print(Dright)

        if(self.useClustering):

            #effettuo clustering
            TsIndexLeft = Dleft['TsIndex']  # TsIndex contenute in Dleft

            CandidatesListLeft = self.OriginalCandidatesListTrain['IdTs'].isin(
                TsIndexLeft)  #  setta a True gli indici dei candidati che sono stati generati dalle Ts contenute in Dleft

            CandidateToCluster = self.OriginalCandidatesListTrain[
                CandidatesListLeft]  # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dleft

            CandidateToCluster = CandidateToCluster.reset_index(drop=True)

            indexChoosenMedoids = reduceNumberCandidates(self, CandidateToCluster,
                                                         returnOnlyIndex=True)  # indici di OriginalCandidatesListTrain conteneti candidati da mantenere

            CandidateToCluster = CandidateToCluster.iloc[indexChoosenMedoids]

            if (verbose):
                print('CANDIDATI RIMASTI IN FIT')
                print(CandidateToCluster)

            #calcolo distanze tra Ts in Dleft e candidati scelti
            Dleft = computeSubSeqDistance(self, TsIndexLeft, CandidateToCluster, self.window_size)



            #RIPETO PER DRIGHT------------------------------------------------------

            TsIndexRight = Dright['TsIndex']  # TsIndex contenute in Dleft

            CandidatesListRight = self.OriginalCandidatesListTrain['IdTs'].isin(
                TsIndexRight) #  setta a True gli indici dei candidati che sono stati generati dalle Ts contenute in Dright

            CandidateToCluster = self.OriginalCandidatesListTrain[
                CandidatesListRight]  # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dleft
            CandidateToCluster = CandidateToCluster.reset_index(drop=True)

            indexChoosenMedoids = reduceNumberCandidates(self,CandidateToCluster,returnOnlyIndex=True)  # indici di OriginalCandidatesListTrain conteneti candidati da mantenere

            CandidateToCluster = CandidateToCluster.iloc[indexChoosenMedoids]

            if (verbose):
                print('CANDIDATI RIMASTI IN FIT')
                print(CandidateToCluster)

            #calcolo distanze tra Ts in Dleft e candidati scelti
            Dright = computeSubSeqDistance(self, TsIndexRight, CandidateToCluster, self.window_size)

            if (verbose):
                print('DLEFT & DRIGHT DOPO IL CLUSTERING')
                print(Dleft)
                print(Dright)


        # chiamata ricorsiva
        if (len(Dleft) > 0):
            self.buildTree(root.left, Dleft, 1, verbose)
        if (len(Dright) > 0):
            self.buildTree(root.right, Dright, 1, verbose)
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
        numAttributes -= 1  # per prendere solo gli attributi utili a xTest
        yTest = testDataset.iloc[:]['class'].values
        yPredicted = np.zeros(len(yTest))
        xTest = testDataset.iloc[:, np.r_[:numAttributes]]

        self.printed = False  # DOPO PRIMA PRINT DEL PERCORSO SETTATO A TRUE, COSI STAMPO UNA VOLTA SOLA

        #dizionario contenete shapelet e informazioni relative
        self.ShapeletDf=pd.DataFrame(columns=['IdShapelet','distance','majorMinor','startingIndex'],index=range(0,len(self.dTreeAttributes)))

        #self.Shapelet=pd.DataFrame(columns=columnList)
        self.Shapelet = pd.DataFrame(columns=['IdShapelet','Shapelet'],index=range(0,len(self.dTreeAttributes)))



        # effettuo predizione per ogni pattern

        for i in range(len(xTest)):
            pattern = xTest.iloc[i]
            if(self.printed==False):
                yPredicted[i] = self.treeExplorerPrint(pattern, root,0)
                if(self.verbose):
                    plotDataAndShapelet(self)
                self.printed=True #dopo la prima stampa, setto a false e smetto di stampare
            else:
                yPredicted[i] = self.treeExplorer(pattern, root) #non stampo piu


        yTest = yTest.astype(int)
        yPredicted = yPredicted.astype(int)

        return yTest, yPredicted

    def treeExplorer(self,pattern, node):
        # caso base, node è foglia
        if (node.value == -1):
            return int(node.data[0])
        else:
            # caso ricorsivo
            attr = int(node.value)
            if (pattern[attr] < node.data[0]):
                return self.treeExplorer(pattern, node.left)
            else:
                return self.treeExplorer(pattern, node.right)

        # setto booleano prima in modo da stampare solo 1 TS

    def treeExplorerPrint(self, pattern, node,counter):
        # caso base, node è foglia
        if (node.value == -1):
            return int(node.data[0])
        else:
            # caso ricorsivo
            self.counter=counter+1 #+1 perche parte da 0 e voglio il numero effettivo
            attr = int(node.value)

            if (self.printed == False):

                idTsShapelet = self.dTreeAttributes[self.dTreeAttributes['IdCandidate'] == int(attr)]["IdTs"].values
                idTsShapelet = idTsShapelet[0]

                startingPosition = self.dTreeAttributes[self.dTreeAttributes['IdCandidate'] == int(attr)][
                    "startingPosition"].values
                startingPosition = startingPosition[0]

                TsContainingShapelet = np.array(self.dfTrain[self.dfTrain['TsIndex'] == idTsShapelet].values)
                TsContainingShapelet = TsContainingShapelet[0]
                TsContainingShapelet = TsContainingShapelet[:len(TsContainingShapelet) - 2]

                if (self.warningDetected):
                    Dp = distanceProfile.naiveDistanceProfile(TsContainingShapelet, int(startingPosition),
                                                              self.window_size, self.TsTestForPrint)
                else:
                    Dp = distanceProfile.massDistanceProfile(TsContainingShapelet, int(startingPosition),
                                                             self.window_size, self.TsTestForPrint)

                val, idx = min((val, idx) for (idx, val) in enumerate(Dp[0]))

                self.ShapeletDf.iloc[counter]['IdShapelet']=attr
                self.ShapeletDf.iloc[counter]['distance'] = val
                self.ShapeletDf.iloc[counter]['startingIndex'] = idx

                self.Shapelet.iloc[counter]['IdShapelet']=attr
                self.Shapelet.iloc[counter]['Shapelet']=TsContainingShapelet[idx:idx+self.window_size]


            if (pattern[attr] < node.data[0]):
                self.ShapeletDf.iloc[counter]['majorMinor'] = -1
                counter += 1
                return self.treeExplorerPrint(pattern, node.left,counter)
            else:
                self.ShapeletDf.iloc[counter]['majorMinor'] = 1
                counter += 1
                return self.treeExplorerPrint(pattern, node.right,counter)





