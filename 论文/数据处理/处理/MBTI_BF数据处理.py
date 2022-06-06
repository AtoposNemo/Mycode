
#import pandas as pd

##去掉数据集author_profiles里的空值和无用列
#def drop_null():
#    df = pd.DataFrame(pd.read_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\数据集\作者信息\author_profiles2.csv',encoding = 'gb2312',))
#    df = df.dropna()
#    df.to_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\数据集\B+.csv',index = False)

############根据作者信息筛选评论
import pandas as pd
import csv

def choose_comment():
    df_a = pd.DataFrame(pd.read_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\数据集\作者信息\B+.csv',encoding = 'utf-8',engine='python'))
    #df_c = pd.DataFrame(pd.read_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\数据集\评论\all_comments_since_2015.csv',encoding = 'utf-8',engine='python'))
    #unicode_escape
    authors = list(df_a['author']) 
    new_author = pd.DataFrame()
    with open(r'E:\vspy\论文相关git\论文数据集\论文初步方案\数据集\评论\all_comments_since_2015.csv','r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader :
            if row[0] in authors:
                row = pd.DataFrame(row)#创建dataframe
                row = pd.DataFrame(row.values.T)#转置
                new_author = new_author.append(row)
    ######找出comment中想要的作者名字保存到新df中
    #author_comment = list(df_c['author'])
    #new_author = pd.DataFrame()
    #for i in range(len(author_comment)):
    #    if author_comment[i] in authors:
    #        new_author = new_author.append(df_c.iloc[i,:])
    #        #print(new_author)        
    new_author.to_csv(r'E:\vspy\论文相关git\论文数据集\论文初步方案\数据集\评论\筛选BF.csv',index = False)
choose_comment()



            
             

