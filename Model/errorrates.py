from editdistance import eval as eval_pred # Levenshtein-Distanz , Levenshtein-distance
import numpy as np

def get_error_rates(gt, pred):
    def get_cer(gt, pred):
        cer = [eval_pred(gt, pred)/len(gt) for gt, pred in zip(gt, pred)]
        return np.mean(cer)

    def get_wer(gt, pred):
        wer = [eval_pred(gt.split(), pred.split())/len(gt.split()) for gt, pred in zip(gt, pred)]
        return np.mean(wer)

    return get_cer(gt, pred), get_wer(gt, pred)