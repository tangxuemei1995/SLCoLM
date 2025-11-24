# from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertPreTrainedModel, BertModel)
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer, BertModel

# BertModel.from_pretrained('bert-base-cased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("/ceph/home/jun01/tangxuemei/pre_trained_models/siku_bert")
tokenizer = BertTokenizer.from_pretrained('/ceph/home/jun01/tangxuemei/pre_trained_models/siku_bert', do_lower_case=True, do_basic_tokenize=True)

text = "我 是 中 国 人"
encoded_input = tokenizer.tokenize(text, return_tensors='pt')
output = model(**encoded_input)
print(output)