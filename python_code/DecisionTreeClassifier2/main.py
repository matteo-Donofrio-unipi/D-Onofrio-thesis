from TestManager2 import executeTest, plotTs, plotTestResults, plotComparison, buildTable,executeShapeletTransform, executeClassicDtree
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

    #plotTestResults(nameFile)

    #fissato il primo su x, vario il secondo su y -> su asse x metti attributo di cui sicuramente nel test hai provato tutti i valori
    max=0
    avg=1
    #plotComparison(nameFile,datasetName,'MaxDepth','WindowSize',max)







if __name__ == "__main__":
    main()
