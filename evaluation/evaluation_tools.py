from os import write
import numpy as np

from time import gmtime, strftime
from evaluation.config import RESULT_DIR
from utils.IO import write_to_file

def extract_dataset_labels(corpus_true_labels):
    """
    Code snippet to correctly format dataset true labels
    """
    res = {}
    for dataset in corpus_true_labels:
        res[dataset] = [corpus_true_labels[dataset][i][1] for i in range(len(corpus_true_labels[dataset]))]
    return res

def extract_res_labels(model_results):
    """
    Code snippet to correctly model results
    """
    res = {}
    for dataset in model_results:        
        res[dataset] = []
        for doc in model_results[dataset]:
            res[dataset].append( ([kp[0] for kp in doc[0]], doc[1])) 
    return res

def evaluate_kp_extraction(model_results, true_labels, model_name: str = "" , save : bool = True) -> None:
    stamp = "{} {}".format(strftime("%Y_%m_%d %H_%M", gmtime()), model_name)
    res = "{}\n ------------- \n".format(stamp)
    res_dic = {}

    for dataset in model_results:
        results_c = { 
                        "Precision" : [], 
                        "Recall" : [],
                        "F1" : []
                    }

        results_kp = {
                        "MAP" : [],
                        "nDCG" : []
                     }

        k_set = [3, 5, 7]
        for k in k_set:
            results_kp["Precision_{}".format(k)] = []
            results_kp["Recall_{}".format(k)] = []
            results_kp["F1_{}".format(k)] = []

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
            for k in k_set:
                p_k = len([kp for kp in top_kp[:k] if kp in true_label]) / float(len(top_kp[:k]))
                r_k = len([kp for kp in top_kp[:k] if kp in true_label]) / len_true_label
                f1_k = 0.0

                if p_k != 0 and r_k != 0:
                    f1_k = ( 2.0 * p_k * r_k ) / ( p_k + r_k)
                
                results_kp["Precision_{}".format(k)].append(p_k)
                results_kp["Recall_{}".format(k)].append(r_k)
                results_kp["F1_{}".format(k)].append(f1_k)

            ap = [ len( [ k for k in top_kp[:p] if k in true_label ] ) / float( p ) for p in range(1,len(top_kp) + 1) if top_kp[p - 1] in true_label ]
            map = np.sum(ap) / float( len( true_label ) )
            ndcg = np.sum( [ 1.0 / np.log2(p + 1) for p in range(1,len(top_kp) + 1) if top_kp[p - 1] in true_label ] )
            ndcg = ndcg / np.sum( [ 1.0 / np.log2(p + 1) for p in range(1,len(true_label) + 1) ] )

            results_kp["MAP"].append(map)	
            results_kp["nDCG"].append(ndcg)	

        res += "\nResults for Dataset {}\n --- \n".format(dataset)

        # Candidate results, in Precision, Recall and F1-Score
        res += "Candidate Extraction Evalution: \n"
        for result in results_c:
            res += "{} = {:.3f}%\n".format(result, np.mean(results_c[result])*100)

        # KP results, in Precision_k, Recall_k, F1-Score_k, MAP and nDCG for KP
        res += "\nKP Ranking Evalution: \n"
        for result in results_kp:
            res += "{} = {:.3f}%\n".format(result, np.mean(results_kp[result])*100)

        if save:

            res_dic[dataset] = {}
            for (name, dic) in [("candidates", results_c), ("kp", results_kp)]:

                res_dic[name] = {}
                for measure in dic:
                    res_dic[name][measure] = dic[measure]
        
    if save:
        with open("{}/raw/{} raw.txt".format(RESULT_DIR, stamp), "a") as f:
            f.write(res.rstrip())
        write_to_file("{}/struct/{}.txt".format(RESULT_DIR, stamp), res_dic)

    print(res)
    return 