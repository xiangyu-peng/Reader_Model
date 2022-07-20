from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

embedder = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")


def similarity_detect(corpus, query, threshold_upper=0.7, threshold_limit=0.3):
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_score = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_score = cos_score.cpu().numpy()[0]
    if cos_score >= threshold_limit and cos_score <= threshold_upper:
        return True
    return False


def similarity_score(corpus, query):
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_score = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_score = cos_score.cpu().numpy()[0]
    return cos_score

    # We use np.argpartition, to only partially sort the top_k results
    # top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    # print("\n\n======================\n\n")
    # print("Query:", query)
    # print("\nTop 5 most similar sentences in corpus:")

    # for idx in top_results[0:top_k]:
    # for idx in range(5):
    #     print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))

def similarity_filter(corpus, queries, threshold=0.8):
    '''
    filter querys when sim to corpus is toohing
    :param corpus:
    :param query:
    :return:
    '''
    check = []
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(queries, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)
    cos_scores = cos_scores.cpu()
    for i, query in enumerate(queries):
        check.append(False if [i for i in cos_scores[i].numpy().tolist() if i > threshold] else True)

    return [q for i, q in enumerate(queries) if check[i]]

def similarity_score_max(corpus, queries, threshold=0.8):
    '''
    filter querys when sim to corpus is toohing
    :param corpus:
    :param query:
    '''
    res = []
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(queries, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)
    cos_scores = cos_scores.cpu()  #query * corpus
    max_score = float(cos_scores.max().numpy())
    if max_score > threshold:
        m = cos_scores.view(cos_scores.size()[0]*cos_scores.size()[1], -1).argmax()
        indices = torch.cat(((m // cos_scores.size()[1]).view(-1, 1), (m % cos_scores.size()[1]).view(-1, 1)), dim=1)
        res = [corpus[int(indices[0][1])], queries[int(indices[0][0])]]
    return max_score, res

if __name__ == "__main__":
    corpus = "fl"
    queries = "Florida"
    # print(
    #     similarity_score(
    #         corpus="Eric ate ice cream more than once a day everyday",
    #         query=[" He ate a cone every day"],
    #     )
    # )
    print(similarity_score_max(corpus=['get fat', 'lose weight'],
                            queries=['get fatty', 'like', 'be fat']))
    # print(similarity_score(corpus=['lose weight'], query=['get fat']))
