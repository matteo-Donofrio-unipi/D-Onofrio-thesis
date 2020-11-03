from TestManager2 import executeTest, plotTs, plotTestResults
from kCandidatesSearch2 import runKMeans
def main():
    datasetName = 'ArrowHead'
    nameFile = datasetName + 'TestResults.csv'
    useValidationSet = False
    usePercentageTrainingSet = True

    executeTest(useValidationSet,usePercentageTrainingSet,datasetName,nameFile)


    #plotTs(datasetName)

    #plotTestResults(nameFile)




if __name__ == "__main__":
    main()
