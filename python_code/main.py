# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ThesisLibrary import *
import time

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

start_time = time.time()

#ACQUISISCO STRUTTURE DATI DEL TRAINING SET
dataset = arff.loadarff('CBF/CBF_TRAIN.arff')
dfTrain=pd.DataFrame(dataset[0])
window_size=5
attributeList=list()
setVariables(5,attributeList)
verbose=True
mpTrain,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain,CandidatesUsedListTrain=getDataStructures(dfTrain,verbose)
dfForDTree=computeSubSeqDistance(dfTrain,CandidatesListTrain)
if(verbose==True):
    print(dfForDTree)

print("--- %s seconds after getting DATA STRUCTURES" % (time.time() - start_time))

#COSTRUISCO DECISION TREE
candidatesGroup=0
albero=None
maxDepth=3
minSamplesLeaf=5
removeUsedCandidate=1
verbose=True
albero=fit(dfForDTree,candidatesGroup,CandidatesUsedListTrain,maxDepth,minSamplesLeaf,numberOfMotifTrain,numberOfDiscordTrain,removeUsedCandidate,verbose)
print(attributeList)
print(albero)
printAll(albero)

print("--- %s seconds after building TREE" % (time.time() - start_time))

#GENERO STRUTTURE DATI PER TEST SET
verbose=True
dataset2 = arff.loadarff('CBF/CBF_TEST.arff')
dfTest = pd.DataFrame(dataset2[0]) #30 record su matrice da 128 attributi + 'b': classe appartenenza
dfTest=dfTest.iloc[50:100] #ne prendo 50 altrimenti impiega tempo troppo lungo, sono 900 record totali

attributeList=sorted(attributeList)
dfForDTreeTest=computeSubSeqDistanceForTest(dfTest,dfTrain,attributeList,CandidatesListTrain,numberOfMotifTrain,numberOfDiscordTrain)
if(verbose==True):
    print(dfForDTreeTest)

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