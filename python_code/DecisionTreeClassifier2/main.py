from TestManager2 import executeTest, plotTs, plotTestResults, plotComparison, buildTable,executeShapeletTransform, executeClassicDtree
from kCandidatesSearch2 import runKMeans
def main():
    datasetName = 'GunPoint'
    nameFile = datasetName + 'TestResults.csv'
    useValidationSet = False
    usePercentageTrainingSet = True

    #executeTest(useValidationSet,usePercentageTrainingSet,datasetName,nameFile)

    #executeShapeletTransform(datasetName)

    executeClassicDtree(datasetName)

    #plotTs(datasetName)

    #plotTestResults(nameFile)

    #plotComparison(nameFile,datasetName,'WindowSize','PercentageTrainingSet')







if __name__ == "__main__":
    main()
