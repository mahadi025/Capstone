# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
import config
import numpy as np
import pickle
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# %%
def load_data_split(splitPath):
    # initialize the data and labels
    data = []
    labels = []
    # loop over the rows in the data split file
    for row in open(splitPath):
        # extract the class label and features from the row
        row = row.strip().split(",")
        label = row[0]
        features = np.array(row[1:], dtype="float")
        # update the data and label lists
        data.append(features)
        labels.append(label)
    # convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    # return a tuple of the data and labels
    return (data, labels)


# derive the paths to the training and testing CSV files
trainingPath = os.path.sep.join([config.BASE_CSV_PATH, "{}.csv".format(config.TRAIN)])
testingPath = os.path.sep.join([config.BASE_CSV_PATH, "{}.csv".format(config.TEST)])
# load the data from disk
print("[INFO] loading data...")
(trainX, trainY) = load_data_split(trainingPath)
(testX, testY) = load_data_split(testingPath)
# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())

# train the model
print("[INFO] training model...")
# model = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(trainX, trainY)
# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(testX)
print(classification_report(testY, preds, target_names=le.classes_))

print("[INFO] performing cross-validation...")
cv_scores = cross_val_score(model, trainX, trainY, cv=3)
print("Cross-validation scores:", cv_scores)

# serialize the model to disk
print("[INFO] saving model...")
f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()


# %%
def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
    # open the input file for reading
    f = open(inputPath, "r")
    # loop indefinitely
    while True:
        # initialize our batch of data and labels
        data = []
        labels = []
        # keep looping until we reach our batch size
        while len(data) < bs:
            # attempt to read the next row of the CSV file
            row = f.readline()
            # check to see if the row is empty, indicating we have
            # reached the end of the file
            if row == "":
                # reset the file pointer to the beginning of the file
                # and re-read the row
                f.seek(0)
                row = f.readline()
                # if we are evaluating we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file
                if mode == "eval":
                    break
            # extract the class label and features from the row
            row = row.strip().split(",")
            label = row[0]
            label = to_categorical(label, num_classes=numClasses)
            features = np.array(row[1:], dtype="float")
            # update the data and label lists
            data.append(features)
            labels.append(label)
        # yield the batch to the calling function
        yield (np.array(data), np.array(labels))


# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())
# derive the paths to the training, validation, and testing CSV files
trainPath = os.path.sep.join([config.BASE_CSV_PATH, "{}.csv".format(config.TRAIN)])
valPath = os.path.sep.join([config.BASE_CSV_PATH, "{}.csv".format(config.VAL)])
testPath = os.path.sep.join([config.BASE_CSV_PATH, "{}.csv".format(config.TEST)])
# determine the total number of images in the training and validation
# sets
totalTrain = sum([1 for l in open(trainPath)])
totalVal = sum([1 for l in open(valPath)])
# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)

# construct the training, validation, and testing generators
trainGen = csv_feature_generator(
    trainPath, config.BATCH_SIZE, len(config.CLASSES), mode="train"
)
valGen = csv_feature_generator(
    valPath, config.BATCH_SIZE, len(config.CLASSES), mode="eval"
)
testGen = csv_feature_generator(
    testPath, config.BATCH_SIZE, len(config.CLASSES), mode="eval"
)

# define our simple neural network
model = Sequential()
model.add(Dense(256, input_shape=(7 * 7 * 2048,), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(len(config.CLASSES), activation="softmax"))

# compile the model
opt = SGD(learning_rate=1e-3, momentum=0.9, decay=1e-3 / 25)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training simple network...")
H = model.fit(
    x=trainGen,
    steps_per_epoch=totalTrain // config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // config.BATCH_SIZE,
    epochs=25,
)
# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print("[INFO] evaluating network...")
predIdxs = model.predict(x=testGen, steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabels, predIdxs, target_names=le.classes_))
# %%
