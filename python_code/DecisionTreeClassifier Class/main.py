from TestManager import executeTest, plotTs, plotTestResults
from kCandidatesSearch import runKMeans
def main():
    datasetName = 'ElectricDevices'
    nameFile = datasetName + 'TestResults.csv'
    useValidationSet = True
    usePercentageTrainingSet = False

    executeTest(useValidationSet,usePercentageTrainingSet,datasetName,nameFile)


    #plotTs(datasetName)

    #plotTestResults(nameFile)




if __name__ == "__main__":
    main()

