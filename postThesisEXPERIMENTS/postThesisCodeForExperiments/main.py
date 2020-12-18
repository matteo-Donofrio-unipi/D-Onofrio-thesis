from TestManager import executeTestTSCMP, buildTable, executeShapeletTransform, executeClassicDtree, \
    executeDecisionTreeStandard, executeKNN
from PlotLibrary import plotComparisonMultiple,plotTs, plotTestResults, plotComparisonSingle
def main():
    datasetName = 'Wine'
    nameFile = datasetName + 'TestResults.csv'
    useValidationSet = False
    usePercentageTrainingSet = True


    initialWS=20
    for i in range(10):
        executeTestTSCMP(useValidationSet,usePercentageTrainingSet,datasetName,nameFile,initialWS)
        initialWS+=10


    #executeTestTSCMP(useValidationSet, usePercentageTrainingSet, datasetName, nameFile)

    #executeShapeletTransform(datasetName)

    #executeClassicDtree(datasetName) #con shapelet

    #executeKNN(datasetName)

    #executeDecisionTreeStandard(datasetName)







    #plotTs(datasetName)

    #plotTestResults(nameFile) #prende in considerazione solo i risultati ottenuti su test set (senza validation)

    #fissato il primo su x, vario il secondo su y
    max=0
    avg=1
    #plotComparisonMultiple(nameFile,datasetName,'MaxDepth','Candidates',max)

    #plotComparisonSingle(nameFile,datasetName,'NumClusterMedoid',avg,UsePercentageTrainingSet=False)







if __name__ == "__main__":
    main()
