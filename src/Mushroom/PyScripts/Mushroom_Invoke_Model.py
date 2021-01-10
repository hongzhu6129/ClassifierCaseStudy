# Call SK learned Model
import sys
import pickle

#input file name
fileName = sys.stdin.readline().rstrip()
print('File Name Received: ' + fileName)
sys.stdout.flush()
#fileName = "C:\Morphy\Mushroom\SVM_model.sav"

#Load model
model = pickle.load(open(fileName, 'rb'))

#Predict on input parameters
while True:
    #Read parameters
    #print("Please enter 10 parameters:")
    iParam = sys.stdin.readline().rstrip()
    if iParam == "":
        break  #end of prediction
    params = [float(s) for s in iParam.split(',')]
    #predict
    result = model.predict([params])
    print(result[0])
    sys.stdout.flush()
print("Program end.")
sys.stdout.flush()
