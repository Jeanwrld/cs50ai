import csv
import sys
import calendar
import random
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "8"  


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )#ML technique that divides a data set into separate training and testing sets to evaluate how well a model generaliszes to new data
    #validation process that allows you to simulate how your model would perfom with new data

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        0- Administrative, an integer
        1- Administrative_Duration, a floating point number
        2- Informational, an integer
        3- Informational_Duration, a floating point number
        4- ProductRelated, an integer
        5- ProductRelated_Duration, a floating point number
        6- BounceRates, a floating point number
        7- ExitRates, a floating point number
        8- PageValues, a floating point number
        9- SpecialDay, a floating point number
        10- Month, an index from 0 (January) to 11 (December)
        11- OperatingSystems, an integer
        12- Browser, an integer
        13- Region, an integer
        14- TrafficType, an integer
        15- VisitorType, an integer 0 (not returning) or 1 (returning)
        16- Weekend, an integer 0 (if false) or 1 (if true)
    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    def convert_cell(i, value):
        types = {0: int, 1: float, 2: int, 3: float, 4:int, 5:float, 6:float, 7:float, 8:float,
        9:float, 11:int, 12:int, 13:int,14:int
        }
        if i == 10:
            try:
                return list(calendar.month_name).index(value.strip()) - 1
            except ValueError:
                return list(calendar.month_abbr).index(value.strip()[:3]) - 1

        if i == 15:
            return 1 if value == "Returning_Visitor" else 0

        if i == 16:
            return 0 if value == "FALSE" else 1
        
        return types.get(i, str)(value)

       

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        data = []
        for row in reader:
            data.append(
                {"evidence": [convert_cell(i, cell) for i, cell in enumerate(row[:17])],
                "label":"Purchase" if row[17] == "TRUE" else "No Purchase"
                 }
            )

    evidence = [row["evidence"] for row in data]
    labels = [1 if row["label"]=="Purchase" else 0 for row in data]
    return evidence, labels 
    
    


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    #model = svm.SVC()
    #model = Perceptron()
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)


    return model

    
def evaluate(labels, predictions):
    true_positive = 0
    true_negative = 0
    total_positive = 0
    total_negative = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positive += 1
            if predicted == 1:
                true_positive += 1
        else:
            total_negative += 1
            if predicted == 0:
                true_negative += 1

    sensitivity = true_positive / total_positive if total_positive else 0
    specificity = true_negative / total_negative if total_negative else 0

    return sensitivity, specificity

    


if __name__ == "__main__":
    main()
