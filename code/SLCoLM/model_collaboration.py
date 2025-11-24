import json
import random
import requests
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re 
# from easyinstruct import ICLPrompt
# from easyinstruct.utils.api import set_openai_key
import json
import re
import openai
import time
from langconv import *
import os
# Step1: Set your own API-KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    API_KEY = os.getenv("BAIDU_API_KEY")
    SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")
    url = "https://aip.baidubce.com/oauth/2.0/token"
    # SECRET_KEY = 
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))
    
def ERNIE(content):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": content
            }

        ],
        "temperature":0.3,
        "response_format": "json_object"
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    return response.text

def gpt_1106(prompt):

    ''' 调用gpt3.5-instruct-1106 0.002$/1K token'''
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106", 
        messages=[
              {"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={ "type": "json_object" }
            )
    result = completion["choices"][0][ "message"]["content"]
    return result

def tongji():
    #train 111193
    # 3208
    #test
    # 14943
    # 401
    # 14857
    # 410
    f = open('/Users/tangtang/Desktop/毕业论文/Expeirments/data/第六章第二部分/dev.json')
    text = f.read()
    train = json.loads(text)
    total_character = 0
    total_data = 0 
    for key in train.keys():
        total_character += len(key)
        total_data += 1

    print(total_character)
    print(total_data)

'''补充实验，demonstration的数量变化的影响'''
'''先计算一下数据的长度'''

def test_data():
    max_length = 0
    output_format = open("./supericl_4/json_format.txt", "r", encoding='utf-8')
    json_format = output_format.read()
    prompt = "./supericl_4/super_icl_ex_4.json"
    test = open(prompt, 'r',encoding='utf-8')
    test = test.read().split('\n')
    
    for i in range(len(test)):
        # if i < 329:
        #     continue
        print(i)
        line = test[i]
        line = line.strip()
        d = json.loads(line)
        length = len(d["prompt"])
        prompt = d["prompt"]
        prompt = prompt.replace("最终输出格式为json",json_format)

        # gpt_result = gpt_1106(prompt)

        # gpt_result = json.loads(gpt_result)

        # output = {'tokens': d['tokens'], 'prompt':prompt,'gpt': gpt_result}
        
        # with open('./supericl_4/k=4/gpt_all.json', 'a+', encoding='utf-8') as json_file:
        #     json.dump(output, json_file, ensure_ascii=False)
        #     json_file.write('\n') 

        # output = {'tokens': d['tokens'],'gpt': gpt_result}
        # with open('./supericl_4/k=4/gpt_only_generation.json', 'a+', encoding='utf-8') as json_file:
        #     json.dump(output, json_file, ensure_ascii=False)
        #     json_file.write('\n') 

        ernie_result = ERNIE(prompt)
        ernie_result = json.loads(ernie_result)
        ernie_result = ernie_result["result"]

        try:
            ernie_result = json.loads(ernie_result)
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ERNIE':ernie_result}
            with open('./supericl_4/k=4/ERNIE_all.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ERNIE': ernie_result}
            with open('./supericl_4/k=4/ERNIE_only_generation.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

        except:
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ERNIE':ernie_result}
            with open('./supericl_4/k=4/ERNIE_all_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ERNIE': ernie_result}
            with open('./supericl_4/k=4/ERNIE_only_generation_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n')

        if length > max_length:
            max_length = length

    print(max_length)

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
    # print(rel_)
    for key in rel_.keys():
        if rel_[key]['gold'] != 0:
            true_label_number += 1
            cor += rel_[key]['cor']
            pre += rel_[key]['pred']
            gold += rel_[key]['gold']
            writer.write(key +'\t')
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
            print("这些关系类型在标准答案中不存在", key)
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


def eval_spuerICL_ex_gpt(path='./supericl_4/k=4/gpt_only_generation.json', test_path='/Users/tangtang/Desktop/毕业论文/Expeirments/第六章实验第二部分/SLMG4LLM/zero_plm(ner+re)_llm+ex/chisre_test.json'):
    '''评价再SUPERICL上增加了关系解释的效果'''
    etag2char = {'PER':'人物','LOC':'地点','OFI':'职官','TIME':'时间','BOOK':'书籍','GPE':'团体'}

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
    
    ner_, rel_ = {'人物':{'pred':0,'gold':0,'cor':0},'时间':{'pred':0,'gold':0,'cor':0},'地点': {'pred':0,'gold':0,'cor':0}, '职官':{'pred':0,'gold':0,'cor':0}, '书籍': {'pred':0,'gold':0,'cor':0}, '团体': {'pred':0,'gold':0,'cor':0}}, {}
    for line in open(path, encoding='utf-8'):
        
        pre = json.loads(line)

        # '''读wx的预测结果'''
        # rel_pre_wx = pre['rel_pred_wx']
        
        # print(rel_pre_wx.replace('\n','').replace(' ',''))
        # rel_pre_wx = rel_pre_wx.replace('\n','').replace("'",'"').replace('真实实体','实体').replace('真实的实体','实体').replace('真实关系','关系').replace('真实关系','关系')

        # group = re.findall('```json(.*?)```', rel_pre_wx)
        # # print(rel_pre_wx)
        # pre_wx = json.loads(group[0])
        # ner_wx, rel_wx = pre_wx['实体'], pre_wx['关系']
        # print(ner_wx,rel_wx)
        '''读gpt的结果'''
        # print(pre['tokens'])

        rel_pre_wx = pre['ds']
        '检查一些不合格的样本'
        if "真实的实体" not in rel_pre_wx.keys():
            print(rel_pre_wx)
            exit()

        ner_wx = rel_pre_wx["真实的实体"]
        rel_wx = rel_pre_wx["真实的关系"]
        '''对rel中的关系进行整理'''
        ori_rel = rel_wx
        rel_wx = {}
        # print(ori_rel)
        # exit()
        for key in ori_rel.keys():
            key_ = key.split("&")[0].replace('（','').replace('）','').replace(' ','').replace("其他关系",'其他')
            if key != key_:
                rel_wx[key_] = []
                if key_ in ori_rel.keys():
                    '''如果有多个关系类型，将他们合并，key_是标准化的关系类型'''
                    rel_wx[key_] += ori_rel[key_]
                    rel_wx[key_] += ori_rel[key]  
                    # print(rel_wx)
                    # print(ori_rel)
                    # exit()
                else:

                    rel_wx[key_] = ori_rel[key]
                    # print(rel_wx)
                    # print(ori_rel)
                    # exit()
            else:
                rel_wx[key] = ori_rel[key]
        
        # print(rel_wx)
        # exit()
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

        # print(gold_re)
        # print(gold_ner)
        # # exit()
        '''
        对大模型预测的结果做整理
        '''
        '''找到模型预测的所有实体'''
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
        
        
        '''融合方式3。将小模型中F值最低的10种关系换成大模型预测的结果'''

        # f.write('slm:'+ '\t' + str(rel_plm) + '\n')
        # f.write('yGPT:' + '\t' + str(rel_gpt) + '\n')
        # f.write('ywx:' + '\t' + str(rel_wx) + '\n')
        need_change_relation = ['救援','祖孙','离开','隶属于','母','归附','死于时间','旧臣','害怕','死于地点']
        
        for key in all_plm_pre[token].keys(): 
            if key not in need_change_relation: #保留哪些在need_change_relation 中关系类型的大模型预测结果
                # if key not in rel_wx.keys():
                rel_wx[key] = all_plm_pre[token][key]
            # else:
                #在以上need_change_relation中的关系类型
        
        # f.write('GPT:' + '\t' + str(rel_gpt) + '\n')
        # f.write('wx:' + '\t' + str(rel_wx) + '\n')
        # f.write('gold:' + '\t' + str(gold_re) + '\n')

        # '''去除实体相同，实体数不等于2，或者实体不在模型预测列表中的样本'''
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
    #     '''评价文心一言还是GPT'''
        rel = rel_wx

        ner = ner_wx
       

        '''计算关系，rel=rel_WX时评估文心一言和小模型协作的效果，rel=rel_GPT时评估GPT'''

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
    # print(rel_)
    # print(ner_)
    evaluation(rel_, path='./supericl_4/gpt_ex_re_metric.txt')
    # evaluation(ner_evaluation, path='./supericl_4/gpt_ex_ner_metric.txt')

    columns = ['Text','gold_ner','gold_re','wx_ner','gpt_re']

    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./supericl_4/result_ex_gpt.csv', index=False, encoding='utf-8-sig')


def deepseek(system_prompt="", user_prompt=""):
    client = OpenAI(api_key="sk-3affaa0c8b08466db1cf5bd9d0e1fc50", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        response_format={
        'type': 'json_object'
    }
    )

    return response.choices[0].message.content



def qwen(system_prompt="", user_prompt=""):
    
    client = OpenAI(
    api_key="sk-9a66a941944a42b4a8422a0118708359",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
    completion = client.chat.completions.create(
        model="qwen-max-2024-09-19",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"enable_thinking": False},
        response_format={"type": "json_object"},
        temperature=0,
        top_p=0,
        # top_k=1,
        # do_sample=False,
        stream=False,
    )
    # full_content = ""
    # # print("流式输出内容为：")
    # for chunk in completion:
    #     # 如果stream_options.include_usage为True，则最后一个chunk的choices字段为空列表，需要跳过（可以通过chunk.usage获取 Token 使用量）
    #     if chunk.choices:
    #         delta = chunk.choices[0].delta
    #         if delta and hasattr(delta, "content") and delta.content:
    #             full_content += delta.content

    # print(full_content)
    # print(completion.choices[0].message.content)
    # print(completion.model_dump_json())
    # exit()
    return completion.choices[0].message.content


def supply_deepseek_icl_slcolm(model="deepseeek"):
    '''补充在deepseek-V3上的实验'''

    ''''ICL + SLCoLM'''
    max_length = 0 
    output_format = open("./supericl_4/json_format.txt", "r", encoding='utf-8')
    json_format = output_format.read()
    index = 0 
    data = []
    save_path = model +"/icl+slcolm/only_generation.json"
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
   
    finished = []
    for line in data:
        line = json.loads(line)
        finished.append(line["tokens"])
    for line in open("./supericl/superICL_ex_gpt.json"):
        # index += 1
        # if index < 130:
        #     continue
        d = json.loads(line)
        if d["tokens"] in finished:
            print("已经完成的样本", d["tokens"])
            continue

        prompt = d["prompt"]
        length = len(prompt)
        if model == "qwen":
            prompt = prompt.replace("最终输出格式为json",'')
            prompt = prompt + "\n" + json_format
        elif model == "deepseek":
            prompt = prompt.replace("最终输出格式为json",json_format)
        # print(prompt)
        # exit()
    
        if model == "qwen":
            ds_result = qwen(user_prompt=prompt)
        elif model == "deepseek":
            ds_result = deepseek(user_prompt=prompt)
        # ds_result = json.loads(ds_result)
        # print(ds_result)
        # ds_result = ds_result["result"]

        try:
            ds_result = json.loads(ds_result)
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ds':ds_result}
            with open('./' + model +'/icl+slcolm/all.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ds': ds_result}
            with open('./' + model +'/icl+slcolm/only_generation.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

        except:
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ds':ds_result}
            with open('./' + model +'/icl+slcolm/all_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ds': ds_result}
            with open('./' + model +'/icl+slcolm/only_generation_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n')

        if length > max_length:
            max_length = length

    # print(max_length)
        

def supply_deepseek_icl(model="deepseek"):

    ''''ICL'''
    max_length = 0 
    output_format = open("./supericl_4/json_format.txt", "r", encoding='utf-8')
    json_format = output_format.read()
    index = 0 
    data = []
    save_path = model + "/icl/only_generation.json"
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
   
    finished = []
    for line in data:
        line = json.loads(line)
        finished.append(line["tokens"])

    for line in open("./supericl/superICL_ex_wx.json"):
        # index += 1
        # if index < 130:
        #     continue
        d = json.loads(line)
        if d["tokens"] in finished:
            print("已经完成的样本", d["tokens"])
            continue

        prompt = d["prompt"]
        length = len(prompt)
        if model == "qwen":
            prompt = prompt.replace("最终输出格式为json",'')
            prompt = prompt + "\n" + json_format
        elif model == "deepseek":
            prompt = prompt.replace("最终输出格式为json",json_format)
        # print(prompt)
        prompt =  prompt.split('\n')
        new_prompt = []
        # print(prompt)
        # exit()
        for sen in prompt:
            if sen.startswith("模型预测") or sen.startswith("给你一些关系类型"):
                continue
            elif "0" in sen:
                continue
            else:
                # print(sen)
                new_prompt.append(sen)

        prompt = "\n".join(new_prompt).replace("现在请你在模型预测的结果上进行修改和补充，", "").replace("根据以上示例和关系解释，","根据以上示例，").replace("模型预测的实体如下，格式为：","").replace("模型预测的关系如下，格式为：","")
        # print(prompt)
        # exit()
        if model == "qwen":
            ds_result = qwen(user_prompt=prompt)
        elif model == "deepseek":
            ds_result = deepseek(user_prompt=prompt)

        try:
            ds_result = json.loads(ds_result)
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ds':ds_result}
            with open('./' + model + '/icl/all.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ds': ds_result}
            with open('./' + model + '/icl/only_generation.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

        except:
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ds':ds_result}
            with open('./' + model + '/icl/all_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ds': ds_result}
            with open('./' + model + '/icl/only_generation_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n')

        if length > max_length:
            max_length = length


def supply_deepseek_zero(model="deepseek"):
    ''''zero-shot'''
    max_length = 0 
    output_format = open("./supericl_4/json_format.txt", "r", encoding='utf-8')
    json_format = output_format.read()
    index = 0 
    data = []
    save_path = model + "/zero/only_generation.json"
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
   
    finished = []
    for line in data:
        line = json.loads(line)
        finished.append(line["tokens"])

    for line in open("./supericl/superICL_ex_wx.json"):
        # index += 1
        # if index < 130:
        #     continue
        d = json.loads(line)
        if d["tokens"] in finished:
            print("已经完成的样本", d["tokens"])
            continue

        prompt = d["prompt"]
        length = len(prompt)
        if model == "qwen":
            prompt = prompt.replace("最终输出格式为json",'')
            prompt = prompt + "\n" + json_format
        elif model == "deepseek":
            prompt = prompt.replace("最终输出格式为json",json_format)
        # print(prompt)
        prompt =  prompt.split('\n')
        new_prompt = []
        # print(prompt)
        # exit()
        for sen in prompt:
            if sen.startswith("模型预测") or sen.startswith("给你一些关系类型"):
                continue
            elif "0" in sen:
                continue
            else:
                # print(sen)
                new_prompt.append(sen)
        new_prompt = new_prompt[10:] #去掉demonstration
        prompt = "\n".join(new_prompt).replace("现在请你在模型预测的结果上进行修改和补充，", "").replace("根据以上示例和关系解释，","").replace("模型预测的实体如下，格式为：","").replace("模型预测的关系如下，格式为：","")
        # print(prompt)
        # exit()
        if model == "qwen":
            ds_result = qwen(user_prompt=prompt)
        elif model == "deepseek":
            ds_result = deepseek(user_prompt=prompt)

        try:
            ds_result = json.loads(ds_result)
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ds':ds_result}
            with open('./' + model + '/zero/all.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ds': ds_result}
            with open('./' + model + '/zero/only_generation.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

        except:
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ds':ds_result}
            with open('./' + model + '/zero/all_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ds': ds_result}
            with open('./' + model + '/zero/only_generation_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n')

        if length > max_length:
            max_length = length


def supply_deepseek_zero_slcolm(model="deepseek"):
    ''''zero-shot'''
    max_length = 0 
    output_format = open("./supericl_4/json_format.txt", "r", encoding='utf-8')
    json_format = output_format.read()
    index = 0 
    data = []
    save_path = model + "/zero+slcolm/only_generation.json"
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
   
    finished = []
    for line in data:
        line = json.loads(line)
        finished.append(line["tokens"])

    for line in open("./supericl/superICL_ex_wx.json"):
        # index += 1
        # if index < 130:
        #     continue
        d = json.loads(line)
        if d["tokens"] in finished:
            print("已经完成的样本", d["tokens"])
            continue

        prompt = d["prompt"]
        length = len(prompt)
        if model == "qwen":
            prompt = prompt.replace("最终输出格式为json",'')
            prompt = prompt + "\n" + json_format
        elif model == "deepseek":
            prompt = prompt.replace("最终输出格式为json",json_format)
        # print(prompt)
        prompt =  prompt.split('\n')
        # new_prompt = []
        # print(prompt)
        # exit()
        # for sen in prompt:
        #     if sen.startswith("模型预测") or sen.startswith("给你一些关系类型"):
        #         continue
        #     elif "0" in sen:
        #         continue
        #     else:
        #         # print(sen)
                # new_prompt.append(sen)
        new_prompt = prompt[18:] #去掉demonstration
        prompt = "\n".join(new_prompt).replace("根据以上示例和关系解释，","根据以上关系解释，").replace("现在请你在模型预测的结果上进行修改和补充", "现在请你在模型预测的结果(包括对应的概率)上进行修改和补充")
        # print(prompt)
        # exit()
        if model == "qwen":
            ds_result = qwen(user_prompt=prompt)
        elif model == "deepseek":
            ds_result = deepseek(user_prompt=prompt)

        try:
            ds_result = json.loads(ds_result)
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ds':ds_result}
            with open('./' + model + '/zero+slcolm/all.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ds': ds_result}
            with open('./' + model + '/zero+slcolm/only_generation.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

        except:
            output = {'tokens': d['tokens'], 'prompt':prompt, 'ds':ds_result}
            with open('./' + model + '/zero+slcolm/all_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n') 

            output = {'tokens': d['tokens'], 'ds': ds_result}
            with open('./' + model + '/zero+slcolm/only_generation_Nojson.json', 'a+', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False)
                json_file.write('\n')

        if length > max_length:
            max_length = length



if __name__ == "__main__":
    #data generation
    #icl_slcolm
    supply_deepseek_icl_slcolm(model="deepseek")
    supply_deepseek_icl_slcolm(model="qwen")

    # icl设置
    supply_deepseek_icl(model="qwen")

    #zero-shot设置
    supply_deepseek_zero(model="deepseek")
    supply_deepseek_zero_slcolm(model="deepseek")
