
import json
import pandas as pd
from model_collaboration import gpt_1106,ERNIE

'''对大小模型的结果融合，多种融合方式'''

from langconv import *

etag2char = {'PER':'人物','LOC':'地点','OFI':'职官','TIME':'时间','BOOK':'书籍','GPE':'团体'}


def fan_jian(char):
    '''
    繁简转化
    '''

    jian_char = Converter('zh-hans').convert(char)

    return jian_char

def evaluation(rel_={}, path='./zero_plm(ner+re)_llm+ex/wx_plm_metric.txt'):
    '''评估rel_中的关系，评价结果存储到path'''
    writer = open(path, 'w', encoding='utf-8')
    cor, pre, gold = 0, 0, 0
    pp,rr = 0, 0
    true_label_number = 0
    for key in rel_.keys():
        if rel_[key]['gold'] != 0:
            true_label_number += 1
            cor += rel_[key]['cor']
            pre += rel_[key]['pred']
            gold += rel_[key]['gold']
            writer.write(key + '\t')
            if rel_[key]['pred'] != 0:
                p = rel_[key]['cor']/rel_[key]['pred']
            else:
                p = 0
            pp += p
            if rel_[key]['gold'] != 0 :
                r = rel_[key]['cor']/rel_[key]['gold']
            else:
                r = 0
            rr += r
            if p+r != 0 :
                f = 2*p*r/(p+r)
            else:
                f = 0
            writer.write(str(p) + '\t' + str(r) + '\t' + str(f) + '\n')
        else:
            print(key)
    p = cor / pre 
    r = cor / gold
    f = f = 2 * p * r / (p + r)
    print(p,r,f)
    writer.write('micro\t' + str(p) + '\t' + str(r) + '\t' + str(f) + '\n')
    p = pp / true_label_number
    r = rr / true_label_number
    f = 2 * r * p / (r + p) 
    print(p,r,f) 
    writer.write('macro\t' + str(p) + '\t' + str(r) + '\t' + str(f) + '\n\n') 

def eval_spuerICL_ex_gpt(path='/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/IPM/supericl_4/k=4/gpt_only_generation.json', test_path='/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/zero_plm(ner+re)_llm+ex/chisre_test.json'):
    '''评价再SUPERICL上增加了关系解释的效果，多种融合方式评测
    '''
    non_directio_rel = ['别名','地点别名','攻伐','位于','生于地点','政治奥援','交谈','作者','见面','合作','游历','死于地点'] #一些无方向，或者是可以根据实体类型区分方向的关系
    
    data = []
    df = pd.read_csv('/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/zero_plm(ner)_llm+ex/zero_ex_plm(ner).csv')
    prompt = {}
    for index, row in df.iterrows():
       prompt[row['Text']] = row['prompt']

    '''用于将所有的小模型预测结果加入计算'''
    all_plm_pre = {}
    test_pre_plm = pd.read_csv('/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/test/pre_gold.csv')
    for index, row in test_pre_plm .iterrows():
        # print(row['Text'])
        pre = json.loads(row['Pre_relations'].replace("'",'"'))
        mid = {}
        for key in pre:
            new_key = key
            if "&" in key:
                new_key = key.split('&')[0]
            new_key = new_key.replace('（','').replace('）','')
            mid[new_key] = []
            for x in pre[key]:
                mid[new_key].append(x[0])
        all_plm_pre[row['Text']] = mid


    with open(test_path, 'r') as file:
        '''加载测试数据'''
        data_test = json.load(file) 
    f = open('/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/IPM/supericl_4/gpt_change.txt', 'w', encoding='utf-8')
    ner_, rel_ = {'人物':{'pred':0,'gold':0,'cor':0},'时间':{'pred':0,'gold':0,'cor':0},'地点': {'pred':0,'gold':0,'cor':0}, '职官':{'pred':0,'gold':0,'cor':0}, '书籍': {'pred':0,'gold':0,'cor':0}, '团体': {'pred':0,'gold':0,'cor':0}}, {}
    
    rel_add = {'cor':0, 'add':0}
    test_samples = []
    for line in open(path, encoding='utf-8'):
        
        pre = json.loads(line)

        # '''读wx的预测结果'''
        # print(pre)
        # exit()
        if pre["tokens"] not in test_samples:
            test_samples.append(pre["tokens"])
        else:
            #不在重复处理同一个句子的预测结果
            continue
        rel_wx = pre["gpt"]["真实的关系"]
        rel_wx_ = rel_wx
        rel_wx = {}
        print(rel_wx_)
        '''整理预测的关系类型'''
        for key in rel_wx_.keys():
            r = key.split('&')[0].replace('（','').replace('）','').replace(' ','').replace("其他关系",'其他')
            rel_wx[r] = rel_wx_[key]
            if r != key:
                if r in rel_wx_.keys():
                    rel_wx[r] += rel_wx_[r]
        '''处理'派遣': "[['操', '宫'], '0.9998'], [['邈', '刘翊'], '0.9555'], [['操', '刘翊'], '0.4605']'''
        # for key in rel_wx.keys():
        #     if isinstance(rel_wx[key],str):

        #     for p in rel_wx[key]:
        #         if isinstance(p, str):
        #             rel_wx[key].remove(p)
        # print(rel_wx)
        # exit()
        ner_wx = pre["gpt"]["真实的实体"]

        rel_gpt = {}


        '''找到测试集中对应的样本'''
        token = pre['tokens']

        gold_sample = {}
        for d in data_test:
            if d['tokens'] == token:
                gold_sample = d
        if gold_sample == {}:
            print('在测试集中未找到对应的样本', token)
            continue
        '''读黄金答案'''
        gold_re,gold_ner = {},{}
        for x in gold_sample['relations']:
            r = x['type'].split('&')[0].replace('（','').replace('）','').replace(' ','')
            if r not in gold_re.keys():
                gold_re[r] = []
            if  r not in rel_.keys():
                rel_[r] = {'pred':0,'gold':0,'cor':0}
            head_span = ''.join([fan_jian(xx) for xx in x['head_span']])
            tail_span = ''.join([fan_jian(xx) for xx in x['tail_span']])
            gold_re[r].append([head_span,tail_span])
        for x in gold_sample['entities']:

            type = etag2char[x['type']]

            span = ''.join([fan_jian(c) for c in x['span']])
            if type not in gold_ner.keys():
                gold_ner[type] = []
            gold_ner[type].append(span)

        '''
        对大模型预测的结果做整理
        '''
        '''找到大模型预测的所有实体，用来评测大模型的NER效果，以及过滤三元组'''
        new_ner_wx = {}
        for key in ner_wx.keys():
            new_ner_wx[key] = []
            for xx in ner_wx[key]:
                if isinstance(xx, list):
                    for x in xx:
                        new_ner_wx[key].append(x)
                else:
                    new_ner_wx[key].append(xx)
    
        wx_clean_ner = [] 
        for k in ner_wx.keys():
            wx_clean_ner += ner_wx[k]
        '''
        找到小模型预测所有实体，只评测三元组
        '''
        for line in prompt[token].split('\n'):
            if line.startswith('模型预测的实体：'):
                ner = line.replace('模型预测的实体：','').split(',')
                wx_clean_ner = [] 
                for n in ner:
                    wx_clean_ner.append(n[2::])


        '''使用小模型预测的三元组结果'''
        rel_plm = {}
        for line in prompt[token].split('\n'):
            if line.startswith('模型预测的三元组：'):
                line = line.replace('模型预测的三元组：','')
                rel_plm = json.loads(line)

        '''使用所有小模型预测的结果'''

        rel_plm = all_plm_pre[token]
        # print(rel_plm)
        # exit()
        '''添加小模型的结果，上一行控制是加所有小模型的结果还是只加>0.6的结果'''
        
        '''融合方式1和2
        方式1:大小模型全部加在一起 方式2:大模型全部，加小模型>0.6的部分
        '''

        # print(rel_wx)

        # for key in rel_plm.keys():

        #     if key not in rel_gpt.keys():
        #         rel_gpt[key] = rel_plm[key]
        #     else:
        #         for x in rel_plm[key]:
        #             if x not in rel_gpt[key]:
        #                 rel_gpt[key].append(x)
        #     if key not in rel_wx.keys():
        #         rel_wx[key] = rel_plm[key]
        #     else:
        #         for x in rel_plm[key]:
        #             if x not in rel_wx[key]:
        #                 rel_wx[key].append(x)


        '''
        救援	0.666666667	0.117647059	0.2
        祖孙	0.2	0.1	0.133333333
        离开	0.5	0.076923077	0.133333333
        隶属于	0.333333333	0.043478261	0.076923077
        母	0	0	0
        归附	0	0	0
        死于时间	0	0	0
        旧臣	0	0	0
        害怕	0	0	0
        死于地点	0	0	0
        '''
        '''融合方式3。将小模型中F值最低的10种关系换成大模型预测的结果'''
        f.write(token + '\n')
        f.write('slm:'+ '\t' + str(rel_plm) + '\n')
        f.write('yGPT:' + '\t' + str(rel_gpt) + '\n')
        f.write('ywx:' + '\t' + str(rel_wx) + '\n')
        need_change_relation = ['见面','投靠','救援','祖孙','离开','隶属于','母','归附','死于时间','旧臣','害怕','死于地点']

        for key in rel_plm.keys(): 
            if key not in need_change_relation: #保留哪些在need_change_relation 中关系类型的大模型预测结果
                rel_gpt[key] = rel_plm[key]
                rel_wx[key] = rel_plm[key]
            # else:
                #在以上need_change_relation中的关系类型
        
        f.write('GPT:' + '\t' + str(rel_gpt) + '\n')
        f.write('wx:' + '\t' + str(rel_wx) + '\n')
        f.write('gold:' + '\t' + str(gold_re) + '\n')

        
            
        '''融合方式4，小模型有的以小模型为准，小模型没有的以大模型为准'''
        # for key in rel_wx.keys():
        #     if key in rel_plm.keys():
        #         rel_wx[key] = rel_plm[key]
        # for key in rel_gpt.keys():
        #     if key in rel_plm.keys():
        #         rel_gpt[key] = rel_plm[key]



        '''去除实体相同，实体数不等于2，或者实体不在模型预测列表中的样本'''
        rel = rel_wx
        rel_wx = {}
        # print(rel)
        for key in rel.keys():
            # print(rel[key])
            rel_wx[key] = []
            if rel[key] != []:
                for pair in rel[key]:
                    if isinstance(pair, str) or pair ==None:
                        continue
                    if len(pair) < 2:
                        continue
                    if len(pair) > 2:
                        pair = pair[0:2]
                    pair_ = pair
                    pair = []
                    for entity in pair_:
                        entity = entity.replace('人物','').replace('时间','').replace('地点','').replace('书籍','').replace('职官','').replace('团体','')
                        pair.append(entity)
                    if  pair[0] in wx_clean_ner and pair[1] in wx_clean_ner and pair[0] != pair[1]:
                        rel_wx[key].append(pair) 

        data.append([token, str(gold_ner), str(gold_re), str(ner_wx), str(rel_wx)])
        '''
        找到最终答案在小模型的基础上增加多少对三元组，正确的有多少组
        '''
        # for key in rel_wx.keys():
        #     rel_add['cor'] += len(rel_wx[key])
        # for key in rel_plm.keys():
        #     rel_add['add'] += len(rel_plm[key])
        for key in rel_wx.keys():
            if key in rel_plm.keys():
                for x in rel_wx[key]:
                    if x not in rel_plm[key]:
                        rel_add['add'] += 1
                        if key in gold_re.keys():
                            if x in gold_re[key]:
                                rel_add['cor'] += 1
                                print(x)
                        
            else:
                for x in rel_wx[key]:
                    rel_add['add'] += 1
                    
                    if key in gold_re.keys():
                        if x in gold_re[key]:
                            rel_add['cor'] += 1
                            print(x)
        
                    #     print(key,x)

            
                                

        '''评价文心一言还是GPT'''
        rel = rel_wx
        ner = ner_wx
    #     # print(rel)
    #     # print(ner)
    #     # exit()
        '''
        计算关系，rel=rel_WX时评估文心一言和小模型协作的效果，rel=rel_GPT时评估GPT
        '''

        for key in gold_re.keys():
            if key not in rel_.keys():
                    rel_[key] =  {'pred':0,'gold':0,'cor':0}
            rel_[key]['gold'] += len(gold_re[key])
            if key in rel.keys():
                for x in gold_re[key]:
                    if key in non_directio_rel:#方向
                        if x in rel[key] or [x[1],x[0]] in rel[key]:
                            
                            rel_[key]['cor'] += 1
                    else:
                        if x in rel[key]:
                            rel_[key]['cor'] += 1
        for key in ner.keys():

            if key not in ner_.keys():
                    ner_[key] =  {'pred':0,'gold':0,'cor':0}
            # print(ner[key])
            ner_[key]['pred'] += len(ner[key])

        for key in rel.keys():
            if key not in rel_.keys():
                    rel_[key] =  {'pred':0,'gold':0,'cor':0}
            rel_[key]['pred'] += len(rel[key])
        '''计算ner'''
        for key in ner_.keys():
            if key in gold_ner.keys():
                ner_[key]['gold'] += len(gold_ner[key])
            
            if key in ner.keys():
                if key in gold_ner.keys():
                    for x in gold_ner[key]:
                        if x in ner[key]:
                            ner_[key]['cor'] += 1
                            ner[key].remove(x)
                            # print(ner[key])
        # exit()

       
    print(rel_)
    print(ner_)

    evaluation(rel_, path='./supericl_4/gpt_ex_merge_3_metric.txt')
    # evaluation(ner_, path='./supericl/gpt_ex_ner_metric.txt')
    print(rel_add['cor'],rel_add['add'])
    print(rel_add['cor']/rel_add['add'])
    columns = ['Text','gold_ner','gold_re','wx_ner','gpt_re']

    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./supericl_4/result_ex_gpt.csv', index=False, encoding='utf-8-sig')
    print("共有测试样本", len(test_samples))

def clean_ernie_data(path):
    '''利用GPT清洗ERNIE的结果'''
    for line in open(path):
        line = json.loads(line)
        text = line["ERNIE"]
        prompt = '请你帮我将以下ERNIE的预测结果整理成以下json格式：{"思考过程": "","真实的实体":{},"真实的关系":{}}\n仅输出整理后的json格式，不要输出其他内容。\nERNIE的预测结果:'
        prompt += str(text)
        gpt_output = gpt_1106(prompt)
        ernie_result = json.loads(gpt_output)
        output = {'tokens': line['tokens'], 'prompt': line['prompt'], 'ERNIE':ernie_result}
        with open('./supericl_4/k=4/ERNIE_all.json', 'a+', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False)
            json_file.write('\n') 

        output = {'tokens': line['tokens'], 'ERNIE': ernie_result}
        with open('./supericl_4/k=4/ERNIE_only_generation.json', 'a+', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False)
            json_file.write('\n') 
       
def eval_spuerICL_ex_ERNIE(path='/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/IPM/supericl_4/k=4/ERNIE_only_generation.json', test_path='/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/zero_plm(ner+re)_llm+ex/chisre_test.json'):
    '''评价再SUPERICL上增加了关系解释的效果，多种融合方式评测
    '''
    non_directio_rel = ['别名','地点别名','攻伐','位于','生于地点','政治奥援','交谈','作者','见面','合作','游历','死于地点'] #一些无方向，或者是可以根据实体类型区分方向的关系
    
    data = []
    df = pd.read_csv('/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/zero_plm(ner)_llm+ex/zero_ex_plm(ner).csv')
    prompt = {}
    for index, row in df.iterrows():
       prompt[row['Text']] = row['prompt']

    '''用于将所有的小模型预测结果加入计算'''
    all_plm_pre = {}
    test_pre_plm = pd.read_csv('/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/test/pre_gold.csv')
    for index, row in test_pre_plm .iterrows():
        # print(row['Text'])
        pre = json.loads(row['Pre_relations'].replace("'",'"'))
        mid = {}
        for key in pre:
            new_key = key
            if "&" in key:
                new_key = key.split('&')[0]
            new_key = new_key.replace('（','').replace('）','')
            mid[new_key] = []
            for x in pre[key]:
                mid[new_key].append(x[0])
        all_plm_pre[row['Text']] = mid


    with open(test_path, 'r') as file:
        '''加载测试数据'''
        data_test = json.load(file) 
    f = open('/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/IPM/supericl_4/ERNIE_change.txt', 'w', encoding='utf-8')
    ner_, rel_ = {'人物':{'pred':0,'gold':0,'cor':0},'时间':{'pred':0,'gold':0,'cor':0},'地点': {'pred':0,'gold':0,'cor':0}, '职官':{'pred':0,'gold':0,'cor':0}, '书籍': {'pred':0,'gold':0,'cor':0}, '团体': {'pred':0,'gold':0,'cor':0}}, {}
    
    rel_add = {'cor':0, 'add':0}
    '''ERNIE有些数据重复了'''
    test_samples = []
    for line in open(path, encoding='utf-8'):
        pre = json.loads(line)
        if pre["tokens"] not in test_samples:
            test_samples.append(pre["tokens"])
        else:
            #不在重复处理同一个句子的预测结果
            continue
        # '''读wx的预测结果'''
        # print(pre)
        # exit()
        rel_wx = pre["ERNIE"]["真实的关系"]
        rel_wx_ = rel_wx
        rel_wx = {}
        print(rel_wx_)
        '''整理预测的关系类型'''
        for key in rel_wx_.keys():
            r = key.split('&')[0].replace('（','').replace('）','').replace(' ','').replace("其他关系",'其他')
            rel_wx[r] = rel_wx_[key]
            if r != key:
                if r in rel_wx_.keys():
                    rel_wx[r] += rel_wx_[r]
        '''处理'派遣': "[['操', '宫'], '0.9998'], [['邈', '刘翊'], '0.9555'], [['操', '刘翊'], '0.4605']'''
        # for key in rel_wx.keys():
        #     if isinstance(rel_wx[key],str):

        #     for p in rel_wx[key]:
        #         if isinstance(p, str):
        #             rel_wx[key].remove(p)
        # print(rel_wx)
        # exit()
        ner_wx = pre["ERNIE"]["真实的实体"]

        rel_gpt = {}


        '''找到测试集中对应的样本'''
        token = pre['tokens']

        gold_sample = {}
        for d in data_test:
            if d['tokens'] == token:
                gold_sample = d
        if gold_sample == {}:
            print('在测试集中未找到对应的样本', token)
            continue
        '''读黄金答案'''
        gold_re,gold_ner = {},{}
        for x in gold_sample['relations']:
            r = x['type'].split('&')[0].replace('（','').replace('）','').replace(' ','')
            if r not in gold_re.keys():
                gold_re[r] = []
            if  r not in rel_.keys():
                rel_[r] = {'pred':0,'gold':0,'cor':0}
            head_span = ''.join([fan_jian(xx) for xx in x['head_span']])
            tail_span = ''.join([fan_jian(xx) for xx in x['tail_span']])
            gold_re[r].append([head_span,tail_span])
        for x in gold_sample['entities']:

            type = etag2char[x['type']]

            span = ''.join([fan_jian(c) for c in x['span']])
            if type not in gold_ner.keys():
                gold_ner[type] = []
            gold_ner[type].append(span)

        '''
        对大模型预测的结果做整理
        '''
        '''找到大模型预测的所有实体，用来评测大模型的NER效果，以及过滤三元组'''
        new_ner_wx = {}
        for key in ner_wx.keys():
            new_ner_wx[key] = []
            for xx in ner_wx[key]:
                if isinstance(xx, list):
                    for x in xx:
                        new_ner_wx[key].append(x)
                else:
                    new_ner_wx[key].append(xx)
    
        wx_clean_ner = [] 
        for k in ner_wx.keys():
            wx_clean_ner += ner_wx[k]
        '''
        找到小模型预测所有实体，只评测三元组
        '''
        for line in prompt[token].split('\n'):
            if line.startswith('模型预测的实体：'):
                ner = line.replace('模型预测的实体：','').split(',')
                wx_clean_ner = [] 
                for n in ner:
                    wx_clean_ner.append(n[2::])


        '''使用小模型预测的三元组结果'''
        rel_plm = {}
        for line in prompt[token].split('\n'):
            if line.startswith('模型预测的三元组：'):
                line = line.replace('模型预测的三元组：','')
                rel_plm = json.loads(line)

        '''使用所有小模型预测的结果'''

        rel_plm = all_plm_pre[token]
        # print(rel_plm)
        # exit()
        '''添加小模型的结果，上一行控制是加所有小模型的结果还是只加>0.6的结果'''
        
        '''融合方式1和2
        方式1:大小模型全部加在一起 方式2:大模型全部，加小模型>0.6的部分
        '''

        # print(rel_wx)

        # for key in rel_plm.keys():

        #     if key not in rel_gpt.keys():
        #         rel_gpt[key] = rel_plm[key]
        #     else:
        #         for x in rel_plm[key]:
        #             if x not in rel_gpt[key]:
        #                 rel_gpt[key].append(x)
        #     if key not in rel_wx.keys():
        #         rel_wx[key] = rel_plm[key]
        #     else:
        #         for x in rel_plm[key]:
        #             if x not in rel_wx[key]:
        #                 rel_wx[key].append(x)


        '''
        救援	0.666666667	0.117647059	0.2
        祖孙	0.2	0.1	0.133333333
        离开	0.5	0.076923077	0.133333333
        隶属于	0.333333333	0.043478261	0.076923077
        母	0	0	0
        归附	0	0	0
        死于时间	0	0	0
        旧臣	0	0	0
        害怕	0	0	0
        死于地点	0	0	0
        '''
        '''融合方式3。将小模型中F值最低的10种关系换成大模型预测的结果'''
        f.write(token + '\n')
        f.write('slm:'+ '\t' + str(rel_plm) + '\n')
        f.write('yGPT:' + '\t' + str(rel_gpt) + '\n')
        f.write('ywx:' + '\t' + str(rel_wx) + '\n')
        need_change_relation = ['救援','祖孙','离开','隶属于','母','归附','死于时间','旧臣','害怕','死于地点']
        need_change_relation = [ '兄弟', '地点别名','救援','祖孙','离开','隶属于','母','归附','死于时间','旧臣','害怕','死于地点']

        for key in rel_plm.keys(): 
            if key not in need_change_relation: #保留哪些在need_change_relation 中关系类型的大模型预测结果
                rel_gpt[key] = rel_plm[key]
                rel_wx[key] = rel_plm[key]
            # else:
                #在以上need_change_relation中的关系类型
        
        f.write('GPT:' + '\t' + str(rel_gpt) + '\n')
        f.write('wx:' + '\t' + str(rel_wx) + '\n')
        f.write('gold:' + '\t' + str(gold_re) + '\n')

        
            
        '''融合方式4，小模型有的以小模型为准，小模型没有的以大模型为准'''
        # for key in rel_wx.keys():
        #     if key in rel_plm.keys():
        #         rel_wx[key] = rel_plm[key]
        # for key in rel_gpt.keys():
        #     if key in rel_plm.keys():
        #         rel_gpt[key] = rel_plm[key]



        '''去除实体相同，实体数不等于2，或者实体不在模型预测列表中的样本'''
        rel = rel_wx
        rel_wx = {}
        # print(rel)
        for key in rel.keys():
            # print(rel[key])
            rel_wx[key] = []
            if rel[key] != []:
                for pair in rel[key]:
                    if isinstance(pair, str) or pair ==None:
                        continue
                    if len(pair) < 2:
                        continue
                    if len(pair) > 2:
                        pair = pair[0:2]
                    pair_ = pair
                    pair = []
                    for entity in pair_:
                        entity = entity.replace('人物','').replace('时间','').replace('地点','').replace('书籍','').replace('职官','').replace('团体','')
                        pair.append(entity)
                    if  pair[0] in wx_clean_ner and pair[1] in wx_clean_ner and pair[0] != pair[1]:
                        rel_wx[key].append(pair) 

        data.append([token, str(gold_ner), str(gold_re), str(ner_wx), str(rel_wx)])
        '''
        找到最终答案在小模型的基础上增加多少对三元组，正确的有多少组
        '''
        # for key in rel_wx.keys():
        #     rel_add['cor'] += len(rel_wx[key])
        # for key in rel_plm.keys():
        #     rel_add['add'] += len(rel_plm[key])
        for key in rel_wx.keys():
            if key in rel_plm.keys():
                for x in rel_wx[key]:
                    if x not in rel_plm[key]:
                        rel_add['add'] += 1
                        if key in gold_re.keys():
                            if x in gold_re[key]:
                                rel_add['cor'] += 1
                                print(x)
                        
            else:
                for x in rel_wx[key]:
                    rel_add['add'] += 1
                    
                    if key in gold_re.keys():
                        if x in gold_re[key]:
                            rel_add['cor'] += 1
                            print(x)
        '''评价文心一言还是GPT'''
        rel = rel_wx
        ner = ner_wx

        '''
        计算关系，rel=rel_WX时评估文心一言和小模型协作的效果，rel=rel_GPT时评估GPT
        '''

        for key in gold_re.keys():
            if key not in rel_.keys():
                    rel_[key] =  {'pred':0,'gold':0,'cor':0}
            rel_[key]['gold'] += len(gold_re[key])
            if key in rel.keys():
                for x in gold_re[key]:
                    if key in non_directio_rel:#方向
                        if x in rel[key] or [x[1],x[0]] in rel[key]:
                            
                            rel_[key]['cor'] += 1
                    else:
                        if x in rel[key]:
                            rel_[key]['cor'] += 1
        for key in ner.keys():

            if key not in ner_.keys():
                    ner_[key] =  {'pred':0,'gold':0,'cor':0}
            # print(ner[key])
            ner_[key]['pred'] += len(ner[key])

        for key in rel.keys():
            if key not in rel_.keys():
                    rel_[key] =  {'pred':0,'gold':0,'cor':0}
            rel_[key]['pred'] += len(rel[key])
        '''计算ner'''
        for key in ner_.keys():
            if key in gold_ner.keys():
                ner_[key]['gold'] += len(gold_ner[key])
            
            if key in ner.keys():
                if key in gold_ner.keys():
                    for x in gold_ner[key]:
                        if x in ner[key]:
                            ner_[key]['cor'] += 1
                            ner[key].remove(x)
    
    print(rel_)
    print(ner_)

    evaluation(rel_, path='./supericl_4/ERNIE_ex_merge_3_metric.txt')
    # evaluation(ner_, path='./supericl/gpt_ex_ner_metric.txt')
    print(rel_add['cor'],rel_add['add'])
    print(rel_add['cor']/rel_add['add'])
    columns = ['Text','gold_ner','gold_re','wx_ner','gpt_re']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./supericl_4/result_ex_ERNIE.csv', index=False, encoding='utf-8-sig')
    print("共有测试样本", len(test_samples))

def supply_ERNIE(path="./supericl_4/k=4/ERNIE_all.json"):
    '''ERNIE差几条数据，在这里经过检查后补充上'''
    output_format = open("./supericl_4/json_format.txt", "r", encoding='utf-8')
    json_format = output_format.read()
    prompt = "./supericl_4/super_icl_ex_4.json"
    test = open(prompt, 'r',encoding='utf-8')
    test = test.read().split('\n')
    already_test_samples = []
    for line in open(path, encoding='utf-8'):
        pre = json.loads(line)
        already_test_samples.append(pre["tokens"])
    for line in test:
        line = json.loads(line)
        if line["tokens"] not in already_test_samples:
            print(line["tokens"])
            prompt = line["prompt"]
            prompt = prompt.replace("最终输出格式为json",json_format)
            ernie_result = ERNIE(prompt)
            ernie_result = json.loads(ernie_result)
            ernie_result = ernie_result["result"]
            # ernie_result = json.loads(ernie_result)
            # output = {'tokens': line['tokens'], 'prompt':prompt, 'ERNIE':ernie_result}
            try:
                ernie_result = json.loads(ernie_result)
                output = {'tokens': line['tokens'], 'prompt':prompt, 'ERNIE':ernie_result}
                with open('./supericl_4/k=4/ERNIE_all.json', 'a+', encoding='utf-8') as json_file:
                    json.dump(output, json_file, ensure_ascii=False)
                    json_file.write('\n') 

                output = {'tokens': line['tokens'], 'ERNIE': ernie_result}
                with open('./supericl_4/k=4/ERNIE_only_generation.json', 'a+', encoding='utf-8') as json_file:
                    json.dump(output, json_file, ensure_ascii=False)
                    json_file.write('\n') 

            except:
                output = {'tokens':line['tokens'], 'prompt':prompt, 'ERNIE':ernie_result}
                with open('./supericl_4/k=4/ERNIE_all_Nojson_new.json', 'a+', encoding='utf-8') as json_file:
                    json.dump(output, json_file, ensure_ascii=False)
                    json_file.write('\n') 

                output = {'tokens': line['tokens'], 'ERNIE': ernie_result}
                with open('./supericl_4/k=4/ERNIE_only_generation_Nojson_new.json', 'a+', encoding='utf-8') as json_file:
                    json.dump(output, json_file, ensure_ascii=False)
                    json_file.write('\n')


    
def eval_spuerICL_ex_deepseek(path='/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/IPM/supericl_4/k=4/ERNIE_only_generation.json', mode=3, test_path='/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/zero_plm(ner+re)_llm+ex/chisre_test.json'):
    '''评价再SUPERICL上增加了关系解释的效果，多种融合方式评测
    '''
    non_directio_rel = ['别名','地点别名','攻伐','位于','生于地点','政治奥援','交谈','作者','见面','合作','游历','死于地点'] #一些无方向，或者是可以根据实体类型区分方向的关系
    save_path = '/'.join(path.split('/')[0:-1]) + '/'
    data = []
    df = pd.read_csv('/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/zero_plm(ner)_llm+ex/zero_ex_plm(ner).csv')
    prompt = {}
    for index, row in df.iterrows():
       prompt[row['Text']] = row['prompt']

    '''用于将所有的小模型预测结果加入计算'''
    all_plm_pre = {}
    test_pre_plm = pd.read_csv('/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/test/pre_gold.csv')
    for index, row in test_pre_plm .iterrows():
        # print(row['Text'])
        pre = json.loads(row['Pre_relations'].replace("'",'"'))
        mid = {}
        for key in pre:
            new_key = key
            if "&" in key:
                new_key = key.split('&')[0]
            new_key = new_key.replace('（','').replace('）','')
            mid[new_key] = []
            for x in pre[key]:
                mid[new_key].append(x[0])
        all_plm_pre[row['Text']] = mid


    with open(test_path, 'r') as file:
        '''加载测试数据'''
        data_test = json.load(file) 
    f = open(save_path +'_change.txt', 'w', encoding='utf-8')
    ner_, rel_ = {'人物':{'pred':0,'gold':0,'cor':0},'时间':{'pred':0,'gold':0,'cor':0},'地点': {'pred':0,'gold':0,'cor':0}, '职官':{'pred':0,'gold':0,'cor':0}, '书籍': {'pred':0,'gold':0,'cor':0}, '团体': {'pred':0,'gold':0,'cor':0}}, {}
    
    rel_add = {'cor':0, 'add':0}
    '''ERNIE有些数据重复了'''
    test_samples = []
    for line in open(path, encoding='utf-8'):
        pre = json.loads(line)
        if pre["tokens"] not in test_samples:
            test_samples.append(pre["tokens"])
        else:
            #不在重复处理同一个句子的预测结果
            continue
        # '''读wx的预测结果'''
        # print(pre)
        # exit()
        rel_wx = pre["ds"]["真实的关系"]
        rel_wx_ = rel_wx
        rel_wx = {}
        # print(rel_wx_)
        '''整理预测的关系类型'''
        for key in rel_wx_.keys():
            r = key.split('&')[0].replace('（','').replace('）','').replace(' ','').replace("其他关系",'其他')
            rel_wx[r] = rel_wx_[key]
            if r != key:
                if r in rel_wx_.keys():
                    rel_wx[r] += rel_wx_[r]
        '''处理'派遣': "[['操', '宫'], '0.9998'], [['邈', '刘翊'], '0.9555'], [['操', '刘翊'], '0.4605']'''
        # for key in rel_wx.keys():
        #     if isinstance(rel_wx[key],str):

        #     for p in rel_wx[key]:
        #         if isinstance(p, str):
        #             rel_wx[key].remove(p)
        # print(rel_wx)
        # exit()
        ner_wx = pre["ds"]["真实的实体"]
        rel_gpt = {}


        '''找到测试集中对应的样本'''
        token = pre['tokens']

        gold_sample = {}
        for d in data_test:
            if d['tokens'] == token:
                gold_sample = d
        if gold_sample == {}:
            print('在测试集中未找到对应的样本', token)
            continue
        '''读黄金答案'''
        gold_re,gold_ner = {},{}
        for x in gold_sample['relations']:
            r = x['type'].split('&')[0].replace('（','').replace('）','').replace(' ','')
            if r not in gold_re.keys():
                gold_re[r] = []
            if  r not in rel_.keys():
                rel_[r] = {'pred':0,'gold':0,'cor':0}
            head_span = ''.join([fan_jian(xx) for xx in x['head_span']])
            tail_span = ''.join([fan_jian(xx) for xx in x['tail_span']])
            gold_re[r].append([head_span,tail_span])
        for x in gold_sample['entities']:

            type = etag2char[x['type']]

            span = ''.join([fan_jian(c) for c in x['span']])
            if type not in gold_ner.keys():
                gold_ner[type] = []
            gold_ner[type].append(span)

        '''
        对大模型预测的结果做整理
        '''
        '''找到大模型预测的所有实体，用来评测大模型的NER效果，以及过滤三元组'''
        new_ner_wx = {}
        for key in ner_wx.keys():
            new_ner_wx[key] = []
            for xx in ner_wx[key]:
                if isinstance(xx, list):
                    for x in xx:
                        new_ner_wx[key].append(x)
                else:
                    new_ner_wx[key].append(xx)
    
        wx_clean_ner = [] 
        for k in ner_wx.keys():
            wx_clean_ner += ner_wx[k]
        '''
        找到小模型预测所有实体，只评测三元组
        '''
        for line in prompt[token].split('\n'):
            if line.startswith('模型预测的实体：'):
                ner = line.replace('模型预测的实体：','').split(',')
                wx_clean_ner = [] 
                for n in ner:
                    wx_clean_ner.append(n[2::])


        '''使用小模型预测的三元组结果'''
        rel_plm = {}
        for line in prompt[token].split('\n'):
            if line.startswith('模型预测的三元组：'):
                line = line.replace('模型预测的三元组：','')
                rel_plm = json.loads(line)

        '''使用所有小模型预测的结果'''

    
        # print(rel_plm)
        # exit()
        '''添加小模型的结果，上一行控制是加所有小模型的结果还是只加>0.6的结果'''
        
        '''融合方式1和2
        方式1:大小模型全部加在一起 方式2:大模型全部，加小模型>0.6的部分
        '''

        # print(rel_wx)
        if mode == 1:
            rel_plm = all_plm_pre[token]
            for key in rel_plm.keys():
                if key not in rel_gpt.keys():
                    rel_gpt[key] = rel_plm[key]
                else:
                    for x in rel_plm[key]:
                        if x not in rel_gpt[key]:
                            rel_gpt[key].append(x)
                if key not in rel_wx.keys():
                    rel_wx[key] = rel_plm[key]
                else:
                    for x in rel_plm[key]:
                        if x not in rel_wx[key]:
                            rel_wx[key].append(x)
        elif mode ==2:
            rel_plm_all = {}
            for line in open("/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/IPM/merge_mode/mode2/triple>0.7.json", "r", encoding='utf-8'):
                sample = json.loads(line)
                for key,value in sample.items():
                    rel_plm_all[key] = value 
            if token not in rel_plm_all.keys():
                rel_plm = {}
            else:
                rel_plm = rel_plm_all[token]
            for key in rel_plm.keys():
                if key not in rel_gpt.keys():
                    rel_gpt[key] = rel_plm[key]
                else:
                    for x in rel_plm[key]:
                        if x not in rel_gpt[key]:
                            rel_gpt[key].append(x)
                if key not in rel_wx.keys():
                    rel_wx[key] = rel_plm[key]
                else:
                    for x in rel_plm[key]:
                        if x not in rel_wx[key]:
                            rel_wx[key].append(x)
            # '''融合方式3。将小模型中F值最低的10种关系换成大模型预测的结果'''
        elif mode == 3:
            f.write(token + '\n')
            f.write('slm:'+ '\t' + str(rel_plm) + '\n')
            f.write('yGPT:' + '\t' + str(rel_gpt) + '\n')
            f.write('ywx:' + '\t' + str(rel_wx) + '\n')
            # need_change_relation = ['救援','祖孙','离开','隶属于','母','归附','死于时间','旧臣','害怕','死于地点']
            need_change_relation = [ '兄弟', '地点别名','救援','祖孙','离开','隶属于','母','归附','死于时间','旧臣','害怕','死于地点']

            for key in rel_plm.keys(): 
                if key not in need_change_relation: #保留哪些在need_change_relation 中关系类型的大模型预测结果
                    rel_gpt[key] = rel_plm[key]
                    rel_wx[key] = rel_plm[key]
                # else:
                    #在以上need_change_relation中的关系类型
            
            f.write('GPT:' + '\t' + str(rel_gpt) + '\n')
            f.write('wx:' + '\t' + str(rel_wx) + '\n')
            f.write('gold:' + '\t' + str(gold_re) + '\n')

        
        elif mode == '4':
        #'''融合方式4，小模型有的以小模型为准，小模型没有的以大模型为准'''

            for key in rel_wx.keys():
                if key in rel_plm.keys():
                    rel_wx[key] = rel_plm[key]
            for key in rel_gpt.keys():
                if key in rel_plm.keys():
                    rel_gpt[key] = rel_plm[key]



        '''去除实体相同，实体数不等于2，或者实体不在模型预测列表中的样本'''
        rel = rel_wx
        rel_wx = {}
        # print(rel)
        for key in rel.keys():
            # print(rel[key])
            rel_wx[key] = []
            if rel[key] != []:
                for pair in rel[key]:
                    if isinstance(pair, str) or pair ==None:
                        continue
                    if len(pair) < 2:
                        continue
                    if len(pair) > 2:
                        pair = pair[0:2]
                    pair_ = pair
                    pair = []
                    for entity in pair_:
                        entity = entity.replace('人物','').replace('时间','').replace('地点','').replace('书籍','').replace('职官','').replace('团体','')
                        pair.append(entity)
                    if  pair[0] in wx_clean_ner and pair[1] in wx_clean_ner and pair[0] != pair[1]:
                        rel_wx[key].append(pair) 

        data.append([token, str(gold_ner), str(gold_re), str(ner_wx), str(rel_wx)])
        '''
        找到最终答案在小模型的基础上增加多少对三元组，正确的有多少组
        '''
        for key in rel_wx.keys():
            if key in rel_plm.keys():
                for x in rel_wx[key]:
                    if x not in rel_plm[key]:
                        rel_add['add'] += 1
                        if key in gold_re.keys():
                            if x in gold_re[key]:
                                rel_add['cor'] += 1
                                print(x)
                        
            else:
                for x in rel_wx[key]:
                    rel_add['add'] += 1
                    
                    if key in gold_re.keys():
                        if x in gold_re[key]:
                            rel_add['cor'] += 1
                            print(x)
        '''评价文心一言还是GPT'''
        rel = rel_wx
        ner = ner_wx

        '''
        计算关系，rel=rel_WX时评估文心一言和小模型协作的效果，rel=rel_GPT时评估GPT
        '''

        for key in gold_re.keys():
            if key not in rel_.keys():
                    rel_[key] =  {'pred':0,'gold':0,'cor':0}
            rel_[key]['gold'] += len(gold_re[key])
            if key in rel.keys():
                for x in gold_re[key]:
                    if key in non_directio_rel:#方向
                        if x in rel[key] or [x[1],x[0]] in rel[key]:
                            
                            rel_[key]['cor'] += 1
                    else:
                        if x in rel[key]:
                            rel_[key]['cor'] += 1
        for key in ner.keys():

            if key not in ner_.keys():
                    ner_[key] =  {'pred':0,'gold':0,'cor':0}
            # print(ner[key])
            ner_[key]['pred'] += len(ner[key])

        for key in rel.keys():
            if key not in rel_.keys():
                    rel_[key] =  {'pred':0,'gold':0,'cor':0}
            rel_[key]['pred'] += len(rel[key])
        '''计算ner'''
        for key in ner_.keys():
            if key in gold_ner.keys():
                ner_[key]['gold'] += len(gold_ner[key])
            
            if key in ner.keys():
                if key in gold_ner.keys():
                    for x in gold_ner[key]:
                        if x in ner[key]:
                            ner_[key]['cor'] += 1
                            ner[key].remove(x)
    
    # print(rel_)
    # print(ner_)

    evaluation(rel_, path=save_path + '/merge_3_metric.txt')
    # evaluation(ner_, path='./supericl/gpt_ex_ner_metric.txt')
    print(rel_add['cor'],rel_add['add'])
    print(rel_add['cor']/rel_add['add'])
    columns = ['Text','gold_ner','gold_re','wx_ner','gpt_re']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path + 'result.csv', index=False, encoding='utf-8-sig')
    print("共有测试样本", len(test_samples))   
    


if __name__ == '__main__':
    #评价 deepseek icl_slcolm结果
    eval_spuerICL_ex_deepseek(path='./deepseek/icl+slcolm/only_generation.json')
    eval_spuerICL_ex_deepseek(path='./deepseek/icl/only_generation.json')

  
    #
