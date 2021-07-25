

def gather_data(prediction, label):

    confusion = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0
    }

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if(label[i][j] == 1):
                if(prediction[i][j] == 1):
                    confusion["tp"] += 1
                elif(prediction[i][j] == 0):
                    confusion["fp"] += 1
            elif(label[i][j] == 0):
                if(prediction[i][j] == 1):
                    confusion["fn"] += 1
                elif(prediction[i][j] == 0):
                    confusion["tn"] += 1
        
    return confusion

def metrics(confusion):

    cer = (confusion['fn']+confusion['fp'])/(confusion['tn']+confusion['fn']+confusion['fp']+confusion['tp'])

    recall = confusion['tp']/(confusion['tp'] + confusion['fn'])
    precision = confusion['tp']/(confusion['tp'] + confusion['fp'])

    f_val = 2*precision*recall/(precision+recall)

    return cer, f_val



