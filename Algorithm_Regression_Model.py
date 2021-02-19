import csv
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error


# Loading data
train_data = []
train_groundtruth = []

valid_data = []
valid_groundtruth = []
test_data = []
test_groundtruth = []

with open('Q3_X_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            train_data.append([float(i) for i in row])
            line_count += 1

with open('Q3_Y_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            train_groundtruth.append(float(row[0]))
            line_count += 1


valid_data = train_data[:100]
valid_groundtruth = train_groundtruth[:100]


train_data = train_data[100:]
train_groundtruth = train_groundtruth[100:]



with open('Q3_X_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            test_data.append([float(i) for i in row])
            line_count += 1

with open('Q3_Y_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            test_groundtruth.append(float(row[0]))
            line_count += 1

# looking for the best alpha for Ridge model, which is alpha=11
for alpha in range(0,50):
    RD = linear_model.Ridge(alpha=alpha)
    RD.fit(train_data, train_groundtruth)
    predict = RD.predict(valid_data)
    error = mean_absolute_error(valid_groundtruth,predict)
    print(alpha, error)

# looking for the best alpha for Ridge model, which alpha = 0.05
for alpha in range(0,100, 5):
    a = alpha/100
    LA = linear_model.Lasso(alpha=a)
    LA.fit(train_data, train_groundtruth)
    predict = LA.predict(valid_data)
    error = mean_absolute_error(valid_groundtruth,predict)
    print(a, error)


LS = linear_model.LinearRegression()
LS.fit(train_data, train_groundtruth)
predict = LS.predict(test_data)
error = mean_absolute_error(test_groundtruth,predict)
print("Least squre error:", error)

RD = linear_model.Ridge(alpha=11)
RD.fit(train_data, train_groundtruth)
predict = RD.predict(test_data)
error = mean_absolute_error(test_groundtruth,predict)
print("Ridge regression error:", error)

LA = linear_model.Lasso(alpha=0.05)
LA.fit(train_data, train_groundtruth)
predict = LA.predict(test_data)
error = mean_absolute_error(test_groundtruth,predict)
print("LASSO regression error:", error)
