from TestManager import executeTestTSCMP, buildTable,executeShapeletTransform, executeClassicDtree
from PlotLibrary import plotComparisonMultiple,plotTs, plotTestResults, plotComparisonSingle
def main():
    datasetName = 'ECG200'
    nameFile = datasetName + 'TestResults.csv'
    useValidationSet = False
    usePercentageTrainingSet = True

    executeTestTSCMP(useValidationSet,usePercentageTrainingSet,datasetName,nameFile)

    #executeShapeletTransform(datasetName)

    #executeClassicDtree(datasetName)

    #plotTs(datasetName)

    #plotTestResults(nameFile) #prende in considerazione solo i risultati ottenuti su test set (senza validation)

    #fissato il primo su x, vario il secondo su y
    max=0
    avg=1
    #plotComparisonMultiple(nameFile,datasetName,'MaxDepth','Candidates',max)

    #plotComparisonSingle(nameFile,datasetName,'NumClusterMedoid',avg,UsePercentageTrainingSet=False)







if __name__ == "__main__":
    main()
