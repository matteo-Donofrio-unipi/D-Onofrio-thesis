import csv
from matplotlib.pyplot import errorbar
from matplotlib import pyplot as plt

def readCsv(fileName):
    # csv file name

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(fileName, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

            # get total number of rows
        print("Total no. of rows: %d" % (csvreader.line_num))

    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields))

    #  printing first 5 rows
    print('\nFirst 5 rows are:\n')
    for row in rows:
        print(row)
        # parsing each column of a row
        # for col in row:
        #     print("%10s" % col)


def WriteCsv(fileName,row):
    #fields = ['Candidates', 'Max depth', 'Min samples', 'Window size', 'Remove candi', 'k', '% Training set', 'Accuracy']

    # writing to csv file
    with open(fileName, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the data rows
        csvwriter.writerow(row)


def PlotValues(fileName):
    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(fileName, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        Percentage=list()
        Accuracy=list()
        i=0
        # extracting each data row one by one
        for row in csvreader:
            if(i==0):
                i = 1
                continue
            Percentage.append(row[6])
            Accuracy.append(row[7])
        # get total number of rows
        print("Total no. of rows: %d" % (csvreader.line_num))


    print(Percentage)
    print('')
    print(Accuracy)

    plt.plot(Percentage, Accuracy)  # Create line plot with yvals against xvals
    plt.show()