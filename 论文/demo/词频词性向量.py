import nltk
import pandas as pd
from nltk import word_tokenize,pos_tag
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import brown 
import numpy as np
import gc



txt = pd.read_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\最终数据\txt.csv')
big_five = pd.read_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\最终数据\big_five.csv')
cixing_csv = pd.DataFrame()
cipin_csv = pd.DataFrame()


#要去除的标点和停用词
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
stops=set(stopwords.words("english"))
cipin = pd.DataFrame()


for i in range(len(txt)):
    text = txt['body'][i].lower()
    text_list=nltk.word_tokenize(text)
    text_list=[word for word in text_list if word not in english_punctuations]
    text_list=[word for word in text_list if word not in stops]

    #词性标注,37维词性向量
    data = nltk.pos_tag(text_list)
    #把标注的形容词筛选出来,构造词频词性向量
    cixing = {'CC':0,'CD':0,'DT':0,'EX':0,'FW':0,'IN':0,'JJ':0,'JJR':0,
              'JJS':0,'NN':0,'NNS':0,'NNP':0,'NNPS':0,'PDT':0,'POS':0,'PRP':0,
              'RB':0, 'RBR':0,'RBS':0,'RP':0,'SYM':0,'TO':0,'UH':0,'VB':0,
              'VBD':0,'VBG':0,'VBN':0,'VBP':0,'VBZ':0,'WDT':0,'WP':0,'WRB':0}
    for word in data:
        if  word[1] in cixing.keys():
            cixing[word[1]]+=1
    cixing = pd.DataFrame(cixing.values())
    cixing = pd.DataFrame(cixing.values.reshape(1,32))
    cixing_csv = cixing_csv.append(cixing)

    #大五人格词频向量
    words = pd.DataFrame(data)
    words_agreeableness_p = len(words[words[0].isin(big_five['agreableness_p'])])
    words_openness_p = len(words[words[0].isin(big_five['openness_p'])])
    words_conscientiousness_p= len(words[words[0].isin(big_five['conscientiousness_p'])])
    words_extraversion_p = len(words[words[0].isin(big_five['extraversion_p'])])
    words_neuroticism_p = len(words[words[0].isin(big_five['neuroticism_p'])])
    
    words_agreeableness_n = len(words[words[0].isin(big_five['agreableness_n'])])
    words_openness_n = len(words[words[0].isin(big_five['openness_n'])])
    words_conscientiousness_n= len(words[words[0].isin(big_five['conscientiousness_n'])])
    words_extraversion_n = len(words[words[0].isin(big_five['extraversion_n'])])
    words_neuroticism_n = len(words[words[0].isin(big_five['neuroticism_n'])])
    
    cipin_list = pd.DataFrame([words_agreeableness_p - words_agreeableness_n,
                            words_openness_p - words_openness_n,
                            words_conscientiousness_p - words_conscientiousness_n,
                            words_extraversion_p - words_extraversion_n,
                            words_neuroticism_p - words_neuroticism_n])
    cipin_list = pd.DataFrame(cipin_list.values.reshape(1,5))
    cipin_csv = cipin_csv.append(cipin_list)

    del cixing,text_list,data,words,cipin_list
    gc.collect()
    
    
#结果保存
cixing_csv.to_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\最终数据\词性向量.csv',index = False)
cipin_csv.to_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\最终数据\词频向量.csv',,index = False