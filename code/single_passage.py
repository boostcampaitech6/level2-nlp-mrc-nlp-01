import collections
import argparse
import json
import os

def post_process_voting(doc_scores, path, topk, test_df):
    """document에 대해 independent하게 inference한 prediction들 중 logits과 score를 고려하여 최종 prediction 생성

    Args:
        doc_scores (np.array): document별 retrieval score
        path (str): inference output directory
        topk (int): 몇 개의 문서를 retrieval할 지에 관한 topk
        test_df (pd.DataFrame): test 데이터 DataFrame 
    """    
    test_ids = test_df['id'].tolist()
    nbest_prediction = collections.OrderedDict()
    prediction = collections.OrderedDict()
    
    nbest_hubo = []
    best_hubo = []
    path = '/'.join(path.split('/')[:-1])

    for i in range(topk):
        nbest_path = f'{path}/{i}_pred/{i}_pred_nbest_predictions.json'
        best_path = f'{path}/{i}_pred/{i}_pred.json'
        
        with open(nbest_path, 'r') as json_file:
            json_data = json.load(json_file)
            nbest_hubo.append(json_data)
        with open(best_path, 'r') as json_file:
            json_data = json.load(json_file)
            best_hubo.append(json_data)
        

    for i in range(len(test_ids)):
        id = test_ids[i]
        max_doc_num = None
        max_logits = -200
        
        for j in range(topk):
            pred = nbest_hubo[j][id][0]
            score = (pred['start_logit'] + pred['end_logit'])
            
            if score < 0:
                score = score * (1 - doc_scores[i][j])
            else:
                score = score * doc_scores[i][j]

                
            if max_logits <= score:
                max_doc_num = j
                max_logits = score
                
        nbest_prediction[id] = nbest_hubo[max_doc_num][id]
        prediction[id] = best_hubo[max_doc_num][id]
        
    nbest_file = f'{path}/nbest_predictions.json'
    best_file = f'{path}/predictions.json'
    
    with open(nbest_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(nbest_prediction, indent=4, ensure_ascii=False) + "\n"
        )
    with open(best_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(prediction, indent=4, ensure_ascii=False) + "\n"
        )