from TestManager import executeTestTSCMP, buildTable, executeShapeletTransform, executeClassicDtree, \
    executeDecisionTreeStandard, executeKNN, executeLearningShapelet
from FileManager import WriteCsvComparison
from PlotLibrary import plotComparisonMultiple,plotTs, plotTestResults, plotComparisonSingle
def main():

    DatasetNames=["ArrowHead","BirdChicken","Coffee","Earthquakes", "ECG200",
                   "ECG5000","FaceFour","GunPoint","ItalyPowerDemand","OliveOil","PhalangesOutlinesCorrect",
                   "Strawberry","Trace","TwoLeadECG","Wafer","Wine","Worms","WormsTwoClass","Yoga"]

    useValidationSet = False
    usePercentageTrainingSet = True

    for i in range(len(DatasetNames)):

        datasetName=DatasetNames[i]
        nameFile = datasetName + 'TestResults.csv'

        executeKNN(datasetName)

    #executeTestTSCMP(useValidationSet,usePercentageTrainingSet,datasetName,nameFile,percentage)

    #executeTestTSCMP(useValidationSet, usePercentageTrainingSet, datasetName, nameFile)

        #executeShapeletTransform(datasetName)
        #executeLearningShapelet(datasetName)

        #executeClassicDtree(datasetName) #con shapelet

        #executeKNN(datasetName)

        #executeDecisionTreeStandard(datasetName)


    #plotTs(datasetName)

    #plotTestResults(nameFile,datasetName) #prende in considerazione solo i risultati ottenuti su test set (senza validation)

    #fissato il primo su x, vario il secondo su y
    max=0 #take best accuracy and min time
    avg=1
    min=-1
    #plotComparisonMultiple(nameFile,datasetName,'MaxDepth','Candidates',max)

    #plotComparisonSingle(nameFile,datasetName,'Candidates',max,UsePercentageTrainingSet=False)







if __name__ == "__main__":
    main()
