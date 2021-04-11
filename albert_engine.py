# encoding: utf-8
import torch
from loguru import logger
from transformers import BertTokenizer, AlbertModel


class albert_vector_engine:
    def __init__(self, pretrained_path='./albert_chinese_tiny_pytorch', padding_idx=0):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.model = AlbertModel.from_pretrained(pretrained_path)
        self.padding_idx = padding_idx

    def padding(self, padding_list):
        max_len = 0
        for item in padding_list:
            max_len = max(len(item), max_len)
        for i in range(len(padding_list)):
            while len(padding_list[i]) < max_len:
                padding_list[i].append(self.padding_idx)
        return padding_list

    def generate_vectors(self, sents):
         convert_ids = [self.tokenizer.encode(item) for item in sents]
         padding_ids = self.padding(convert_ids)
         sent_vectors = self.model(torch.tensor(padding_ids))
         # print(sent_vectors[0].shape)
         avg_semantic_vectors = torch.mean(sent_vectors[0], dim=1)
         logger.info("{} vectors generated".format(len(sents)))
         return avg_semantic_vectors

    def cosine(self, vec1, vec2):
         return torch.cosine_similarity(vec1.reshape(-1), vec2.reshape(-1), dim=0)

    def topn(self, torch_vector_list, topn=3):
        if topn > len(torch_vector_list):
            topn = len(torch_vector_list)
        temp = torch.stack(torch_vector_list)
        topn_values, topn_indexs = torch.topk(temp, topn)
        return topn_indexs.detach().numpy().tolist()[:topn], topn_values.detach().numpy().tolist()[:topn]


if __name__ == '__main__':
    vector_engine = albert_vector_engine()
    sents = ["你好", "你好呀", "今天天气", "明天天气", "后天天气怎么样", "后天天气怎么样"]
    vectors = vector_engine.generate_vectors(sents)
    print(vectors, vectors.shape)
    print(vector_engine.cosine(vectors[0], vectors[1]))
    print(vector_engine.cosine(vectors[2], vectors[3]))
    print(vector_engine.cosine(vectors[2], vectors[4]))
    print(vector_engine.cosine(vectors[4], vectors[5]))
