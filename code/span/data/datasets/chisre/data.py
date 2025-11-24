# -- coding: utf-8 --**
import json

from langconv import *
en_type = {'OFI':'职 官','LOC':'地 点','TIME':'时 间','PER':'人 物','GOP':'团 体','GPE':'团 体','BOOK':'书 籍'}

def fan_jian(char):
    '''
    繁简转化
    '''

    jian_char = Converter('zh-hans').convert(char)

    return jian_char

fp = open('./test.json','r',encoding='utf-8')
trip = json.load(fp)
d = []
count = 0
entities_type = {}
relations_type = {}
for key in trip.keys():
    # print(trip[key])

    count += 1
    tokens = [fan_jian(x) for x in key]
    tokens = ''.join(tokens)
    if len(tokens) > 510:
        print(len(tokens))
        print(key)
        continue
    entities = trip[key]['entities']
    for list in entities:
        if list['type'] not in entities_type.keys():
            entities_type[list['type']] = {}
        entities_type[list['type']]['short'] = en_type[list['type']]
        entities_type[list['type']]['verbose'] = en_type[list['type']]
    relations = trip[key]['relations']
    for i in range(len(relations)):
        list = relations[i]
        new_re_type = ' '.join([x for x in list['type']])
        if new_re_type  not in relations_type.keys():
            relations_type[new_re_type ] = {}
        relations[i]['type'] = new_re_type 
        relations_type[list['type']]['short'] = new_re_type 
        relations_type[list['type']]['verbose'] = new_re_type 
        relations_type[list['type']]['symmetric'] = False
    orig_id =  count
    d.append({'tokens':tokens,'entities':entities,'relations':relations,orig_id:'orig_id'})
import random 
# random.shuffle(d)
f =  open('./chisre_test.json','w',encoding='utf-8')
json.dump(d,f,ensure_ascii=False)

# f =  open('./acr_train_dev.json','w',encoding='utf-8')
# json.dump(d[int(count/10):int(count/10)*2],f,ensure_ascii=False)

# f =  open('./acr_train.json','w',encoding='utf-8')
# json.dump(d[int(count/10)*2:],f,ensure_ascii=False)

# f =  open('./triplet_all.json','w',encoding='utf-8')
# json.dump(d,f,ensure_ascii=False)

# f =  open('./acr_types.json','w',encoding='utf-8')
# json.dump({'entities':entities_type,'relations':relations_type},f,ensure_ascii=False)




