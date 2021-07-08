import numpy as np

from time import gmtime, strftime
from evaluation.config import RESULT_DIR

def extract_doc_labels(corpus_true_labels, n):
    res = {}
    for dataset in corpus_true_labels:
        res[dataset] = [corpus_true_labels[dataset][i][1] for i in range(n)]
    return res

def extract_res_labels(model_results):
    res = {}
    for dataset in model_results:        
        res[dataset] = []
        for doc in model_results[dataset]:
            res[dataset].append( ([kp[0] for kp in doc[0]], doc[1])) 
    return res

def evaluate_kp_extraction(model_results, true_labels, model_name: str = "" , write_to_file : bool = True) -> None:
    res = "{} {} \n ------------- \n".format(strftime("%Y_%m_%d %H_%M", gmtime()), model_name)

    for dataset in model_results:
        results_c = { 
                        "Precision" : [], 
                        "Recall" : [],
                        "F1" : []
                    }

        results_kp = {}

        for i in range(len(model_results[dataset])):
            candidates = model_results[dataset][i][1]
            len_candidates = float(len(candidates))

            top_kp = model_results[dataset][i][0]
            len_top_kp = float(len(top_kp))

            true_label = true_labels[dataset][i]
            len_true_label = float(len(true_label))

            # Precision, Recall and F1-Score for candidates
            p = len([kp for kp in candidates if kp in true_label]) / len_candidates
            r = len([kp for kp in candidates if kp in true_label]) / len_true_label
            f1 = 0.0

            if p != 0 and r != 0:
                f1 = ( 2.0 * p * r ) / ( p + r)

            results_c["Precision"].append(p)
            results_c["Recall"].append(r)
            results_c["F1"].append(f1)

            # Precision_k, Recall_k, F1-Score_k, MAP and nDCG for KP



        res += "\nResults for Dataset {}\n --- \n".format(dataset)

        # Candidate results, in Precision, Recall and F1-Score
        res += "Candidate Extraction Evalution: \n"
        for result in results_c:
            res += "{} = {:.3f}%\n".format(result, np.mean(results_c[result])*100)

        # KP results, in Precision_k, Recall_k, F1-Score_k, MAP and nDCG for KP
        res += "\nKP Ranking Evalution: \n\n"
        print(res)
        


    return