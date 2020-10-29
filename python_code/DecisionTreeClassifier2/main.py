from TestManager2 import executeTest, plotTs, plotTestResults
from kCandidatesSearch2 import runKMeans
def main():
    datasetName = 'ECG200'
    nameFile = datasetName + 'TestResults.csv'
    useValidationSet = True
    usePercentageTrainingSet = False

    executeTest(useValidationSet,usePercentageTrainingSet,datasetName,nameFile)


    #plotTs(datasetName)

    #plotTestResults(nameFile)




if __name__ == "__main__":
    main()
