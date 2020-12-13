from TestManager2 import executeTest, plotTs, plotTestResults, plotComparisonSingle, buildTable,executeShapeletTransform, executeClassicDtree,plotComparisonMultiple
from kCandidatesSearch2 import runKMeans
def main():
    datasetName = 'ECG200'
    nameFile = datasetName + 'TestResults.csv'
    useValidationSet = False
    usePercentageTrainingSet = True

    executeTest(useValidationSet,usePercentageTrainingSet,datasetName,nameFile)

    #executeShapeletTransform(datasetName)

    #executeClassicDtree(datasetName)

    #plotTs(datasetName)

    #plotTestResults('ECG200TestResults.csv')

    #fissato il primo su x, vario il secondo su y
    max=0
    avg=1
    #plotComparisonMultiple(nameFile,datasetName,'MaxDepth','Candidates',max)

    #plotComparisonSingle(nameFile,datasetName,'NumClusterMedoid',avg,UsePercentageTrainingSet=True)







if __name__ == "__main__":
    main()
