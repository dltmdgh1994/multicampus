# Pandas

pandas는 numpy를 기본으로 그 위에 Series와 DataFrame이라는 2가지 자료구조를 정의해서 사용

## 1. Series

동일한 데이터 타입의 복수개의 성분으로 구성된 1차원 자료구조

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

s = pd.Series([1,2,3,4,5], dtype=np.float64)
print(s)    # 0    1.0
            # 1    2.0
            # 2    3.0
            # 3    4.0
            # 4    5.0

print(s.values) # [1. 2. 3. 4. 5.]
print(s.index) # RangeIndex(start=0, stop=5, step=1)

s = pd.Series([1,5,10,8],
              dtype=np.int32,
              index=['a','b','b','c'])

print(s['a'])  # 1
print(s[0])    # 1 인덱스가 변경되고 기본적으로 숫자 인덱스는 사용할 수 있어요!

print(s['b']) # b   5
			  # b   10


## Slicing
print(s[0:3]) # slicing 그대로 적용이 되요! 3개의 원소 출력
print(s['a':'c']) # 결과가 달라요! 4개의 원소 출력

# Fancy indexing, boolean indexing
print(s[[0,2]])  # Fancy indexing도 사용가능해요!
				 # a   1
    			 # b   10
print(s[['a','c']])  # Fancy indexing도 사용가능해요!
				 # a   1
				 # c   5
print(s[s%2==0]) # boolean indexing도 가능해요!
				 # b   10
    			 # c   8

# 집계함수 사용 가능
print(s.sum()) # 24

## Series의 사칙연산은 같은 index를 기반으로 수행


## 활용 예시
# A 공장의 2020-01-01부터 10일간 생산량을 Series로 저장
# 생산량은 평균이 50이고 표준편차가 5인 정규분포에서 랜덤하게 생성(정수)
# 형식) 2020-01-01 52
#       2020-01-02 49
#       2020-01-03 55

np.random.seed(1)
start_day = datetime(2020,1,1)
factory_A = pd.Series([int(x) for x in np.random.normal(50,5,(10,))],
                      dtype=np.int32,
                      index=[start_day + timedelta(days=x) 
                             for x in range(10)])

print(factory_A)
#2020-01-01    58
#2020-01-02    46
#2020-01-03    47
#2020-01-04    44
#2020-01-05    54
#2020-01-06    38
#2020-01-07    58
#2020-01-08    46
#2020-01-09    51
#2020-01-10    48
#dtype: int32


## dict을 통해 Series 생성
my_dict = { '서울' : 1000, '인천' : 2000, '수원' : 3000 }
s = pd.Series(my_dict)
print(s) 
#서울    1000
#인천    2000
#수원    3000
#dtype: int64
```



## 2.  DataFrame

Table 형식으로 데이터를 저장하는 자료구조

```python
import numpy as np
import pandas as pd
import pymysql

## pandas에서 DataFrame을 만드는 여러가지 방법
# 1. python의 dict(dictionary)를 직접 작성해서 DataFrame을 생성
my_dict = { 'name' : ['홍길동','신사임당','김연아','강감찬'], 
            'year' : [2015, 2016, 2019, 2016], 
            'point' : [3.5, 1.4, 2.0, 4.5] }

df = pd.DataFrame(my_dict)
print(df.shape) # (4,3)
print(df.size)  # 12
print(df.ndim)  # 2
print(df.index) # RangeIndex(start=0, stop=4, step=1)
print(df.columns) # Index(['name', 'year', 'point'], dtype='object')
print(df.values) # 2차원 numpy
                # [['홍길동' 2015 3.5]
                #  ['신사임당' 2016 1.4]
                #  ['김연아' 2019 2.0]
                #  ['강감찬' 2016 4.5]]
                
# 2. CSV파일을 이용해서 DataFrame을 생성하는 방법
df = pd.read_csv('./movies.csv')

# 3. Database안에 있는 데이터를 이용해서 DataFrame을 생성
# 데이터베이스 접속
# 만약 연결에 성공하면 연결객체가 생성되요!!
conn = pymysql.connect(host='localhost',
                       user='data',
                       password='data',
                       db='library',
                       charset='utf8')

keyword = '파이썬'
sql = "SELECT bisbn, btitle, bauthor, bprice FROM book WHERE btitle LIKE '%{}%'".format(keyword)

try:
    df = pd.read_sql(sql, con=conn)
    display(df)
except Exception as err:    
    print(err)
finally:
    conn.close()
    
# column명을 json의 key값으로 이용해서 JSON을 생성
# with를 사용하면 사용 후 자동으로 close 해준다.
with open('./data/books_orient_column.json', 'w', encoding='utf-8') as file1:
    df.to_json(file1, force_ascii=False, orient='columns')
    
    
# 4. JSON파일을 이용해 DataFrame 생성
df = pd.read_json("books_orient_records.json")


# 5. OpenAPI를 이용해 DataFrame 생성
import json
import urllib

open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json' # Open API URL
query_string = '?key=' #발급 받은 키 값 입력
open_api_url = open_api + query_string

page_obj = urllib.request.urlopen(open_api_url)
json_page = json.loads(page_obj.read())

my_dict = dict()
rank_list = list()
title_list = list()
sales_list = list()

for tmp_dict in json_page['boxOfficeResult']['dailyBoxOfficeList']:
    rank_list.append(tmp_dict['rank'])
    title_list.append(tmp_dict['movieNm'])
    sales_list.append(tmp_dict['salesAmt'])

my_dict['순위'] = rank_list  
my_dict['제목'] = title_list  
my_dict['매출액'] = sales_list  

df = pd.DataFrame(my_dict)
```



### 1. DataFrame Indexing

```python
import numpy as np
import pandas as pd

data = { '이름' : ['홍길동','신사임당','강감찬','아이유','김연아'],
         '학과' : ['컴퓨터', '철학', '수학', '영어영문', '통계'],
         '학년' : [1, 2, 2, 4, 3],
         '학점' : [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data,
                  columns=['학과','이름','학년','학점','등급'],
                  index=['one','two','three','four','five'])

print(df['이름'])   # Series로 출력
stu_name = df['이름']    # View가 나와요!!
stu_name = df['이름'].copy()    # 별도의 Series를 생성

## 원하는 값으로 바꾸고자 할 떄
# 컬럼 기반 처리
df['등급'] = 'A'  # broadcasting => '등급'의 모든 행이 'A'로 입력
df['등급'] = ['A','C','B','A',np.nan] # np.nan => pandas에서 결측치 표현
df['등급'] = ['A','C','B','A'] # 행의 개수가 맞지 않아 오류

# index 기반 처리
df['나이'] = pd.Series([15,30,35],
                    index=['one','three','four']) 


# 연산을 통해서 새로운 column을 추가할 수 있어요!!
df['장학생여부'] = df['학점'] > 4.0


## DataFrame안에 데이터를 삭제하는 경우
new_df = df.drop('학년',axis=1,inplace=False)
# inplac=True => 원본 변경, False => 삭제된 복사본 생성 (defalt : inplace=False)
# axis=0 => 행 삭제, 1 => 열 삭제 (defalt : axis=0)


# column indexng
print(df['이름'])   # OK. Series로 결과 리턴
print(df['이름':'학년'])  # Error. column은 slicing이 안되요!
display(df[['이름','학년']]) # OK. Fancy indexing은 허용!
# boolean indexing은 column과는 상관이 없어요. row indexing할때 사용 
        
    
# Row indexing(숫자 index를 이용해서)
print(df[0])      # Error 행에 대한 숫자 인덱스로 단일 indexing이 안되요!
display(df[1:])    # OK slicing은 가능!   => View
display(df[[1,3]])    # Error Fancy indexing 불가능

# Row indexing(index를 이용해서)
print(df['two'])    # Error. 행에 대한 index를 이용한 단일 row 추출은 안되요!
display(df['two':])  # OK! index를 이용한 row slicing 가능
display(df['two':-1])  # Error! 숫자index와 일반 index를 혼용해서 사용할 수 없어요!
display(df[['one','three']]) # Error  

    
## 혼동스럽기 때문에 row indexing은 loc[]을 사용
# loc[]은 숫자 index를 사용할 수 없고 컬럼명을 이용
display(df.loc['two']) # OK 단일 row 추출 가능 => Series로 리턴
display(df.loc['two':'three']) # OK Slicing
display(df.loc[['two','four']]) # OK Fancy indexing

# 숫자 index를 사용하려면 iloc[]를 이용
display(df.iloc[1]) # OK 단일 row 추출 가능 => Series로 리턴
display(df.iloc[1:3])  # OK Slicing
display(df.iloc[[0,3]])  # OK Fancy indexing

# loc[,]을 통해 행과 열 접근 가능
display(df.loc['one' : 'three', ['이름','학점']])
display(df.loc[df['학점'] > 4.0, ['이름','학점']]) # boolean mask
```



### 2. 메서드

#### 1. 집계함수, 공분산, 상관계수

```python
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime

start = datetime(2019,1,1)  # 2019-01-01 날짜 객체 생성
end = datetime(2019,12,31)  # 2019-12-31 날짜 객체 생성

# YAHOO에서 제공하는 KOSPI 지수
df_KOSPI = pdr.DataReader('^KS11','yahoo',start,end)

# YAHOO에서 제공하는 삼성전자 지수
df_SE = pdr.DataReader('005930.KS','yahoo',start,end)



## numpy가 제공하는 함수를 이용해서 공분산을 계산
print(np.cov(df_KOSPI['Close'].values,df_SE['Close'].values))
# 0행 0열 : KOSPI의 공분산( KOSPI & KOSPI )
# 0행 1열 : KOSPI와 삼성전자의 공분산
# 1행 0열 : 삼성전자와 KOSPI의 공분산
# 1행 1열 : 삼성전자의 공분산( 삼성전자 & 삼성전자 )
# [[6.28958682e+03 9.46863621e+04]
#  [9.46863621e+04 1.41592089e+07]]


## 상관계수를 구하는 함수
my_dict = {
    'KOSPI' : df_KOSPI['Close'],
    '삼성전자' : df_SE['Close'],
}
df = pd.DataFrame(my_dict)
display(df.corr())
# 1.00000  0.31729
# 0.31729  1.00000


## 집계함수
data = [[2, np.nan],
        [7, -3],
        [np.nan, np.nan],
        [1, -2]]

df = pd.DataFrame(data,
                  columns=['one', 'two'],
                  index=['a','b','c','d'])

display(df.sum())   # axis를 생략하면 기본이 axis=0
                    # skipna = True(기본값) 
                    # Series로 리턴!
# one    10.0
# two    -5.0
# dtype: float64

display(df['two'].sum()) # -5.0
display(df.loc['b'].sum()) # 4.0

display(df.mean(axis=0,skipna=False))
# one   NaN
# two   NaN
# dtype: float64
display(df.mean(axis=0,skipna=True))
# one    3.333333
# two   -2.500000
# dtype: float64

# NaN을 df['two']의 평균으로 채워줌
df = df.fillna(value=df['two'].mean())  
```



#### 2. Sort

```python
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0,10,(6,4)))

df.columns = ['A','B','C','D'] # 컬럼명 설정
df.index = pd.date_range('20200101', periods=6) # 인덱스 설정(날짜로)

# permutation()의 특징은 원본은 변경하지 않고 순서가바뀐 복사본을 리턴
new_index = np.random.permutation(df.index)
df2 = df.reindex(index=new_index, columns=['B','A','D','C'])


# shuffle()의 특징은 원본의 바꿔요!
np.random.shuffle(df.index)  


# sort => 원본을 바꾸지 않고 복사본을 리턴
dfv = df2.sort_index(axis=1, ascending=True) # 인덱스를 기준
dfi = df2.sort_values(by=['B','A']) # 값을 기준

DataFrame.sort_values(by=None,  # 정렬할 기준 변수
						axis=0, # 0 : 행(index), 1 : 열(column)
						ascending=True,  # 오름차순 or 내림차순
						inplace=False, # 원본 바꿈 여부
						kind='quicksort', # 정렬 알고리즘
						na_position='last') #결측값 위치 {'first','last'}                     
```



#### 3. Merge

```python
import numpy as np
import pandas as pd

data1 = {
    '학번' : [1,2,3,4],
    '이름' : ['홍길동','신사임당','아이유','김연아'],
    '학년' : [2,4,1,3]
}

data2 = {
    '학번' : [1,2,4,5],
    '학과' : ['컴퓨터','철학','심리','영어영문'],
    '학점' : [3.5, 2.7, 4.0, 4.3]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# how = {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’
# '학번'이라는 컬럼이 공통적으로 존재
display(pd.merge(df1, df2, on='학번', how='inner'))

# '학번'이라는 컬럼이 공통적으로 존재하지만 이름이 다름('학번', '학생학번')
# left_on과 right_on을 사용
display(pd.merge(df1, df2, left_on='학번', right_on='학생학번', how='inner'))

# df2의 학번이 index로 존재
# left_on과 right_index를 사용
result = pd.merge(df1, df2,
                  left_on='학번',
                  right_index=True,
                  how='inner')

# 둘 다 학번이 index로 존재
# left_index와 right_index를 사용
result = pd.merge(df1, df2,
                  left_index=True,
                  right_index=True,
                  how='inner')

DataFrame.merge(df, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
```



#### 4. Concatenation

```python
import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.arange(6).reshape(3,2),
                   index=['a','b','d'],
                   columns=['one','two'])

df2 = pd.DataFrame(np.arange(4).reshape(2,2),
                   index=['a','c'],
                   columns=['three','four'])

result = pd.concat([df1,df2],
                   axis=0,
                   ignore_index=True) # index 번호를 무시하고 새로 메김
```



#### 5. Grouping

```python
import numpy as np
import pandas as pd

my_dict = {
    '학과' : ['컴퓨터','경영학과','컴퓨터','경영학과','컴퓨터'],
    '학년' : [1, 2, 3, 2, 3],
    '이름' : ['홍길동','신사임당','김연아','아이유','강감찬'],
    '학점' : [1.5, 4.4, 3.7, 4.5, 4.2]
}

df = pd.DataFrame(my_dict)

# 학과를 기준으로 grouping
score = df['학점'].groupby(df['학과'])

# 그룹안에 데이터를 확인하고 싶은 경우에는 get_group()
print(score.get_group('경영학과'))

# grouping된 객체에 다양한 함수 적용
print(score.size())  # Series로 리턴되요!
print(score.mean())  
print(score.count())


# 학과,학년를 기준으로 grouping
score = df['학점'].groupby([df['학과'],df['학년']])
print(score.mean())  # 결과가 Series로 나와요(멀티인덱스)
display(score.mean().unstack())  # 최하위 index를 column으로 변경


# 1. 학과별 평균학점은?
df.groupby(df['학과'])['학점'].mean()
# 2. 학과별 몇명이 존재하나요?
df.groupby(df['학과'])['이름'].count()

for dept, group in df.groupby(df['학과']):
    print(dept)
    display(group)
```





#### 6. 기타 함수

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,10,(6,4)))
df.columns = ['A','B','C','D']
df.index = pd.date_range('20200101', periods=6)
df['E'] = ['AA','BB','CC','CC','AA','CC']
display(df)

print(df['E'].unique())   # ['AA' 'BB' 'CC']
print(df['E'].value_counts())  # 각 value값들의 개수를 series로 리턴
print(df['E'].isin(['AA','BB']))  # 조건을 검색할때 많이 이용하는 방법 중 하나
								  # True, False를 series로 리턴
    

new_df = df.dropna(how='any')  
# how='any' NAN이 하나라도 해당 행에 존재하면 행을 삭제
# how='all' 모든 컬럼의 값이 NaN인 경우 행을 삭제
# inplace = False로 default 설정 

new_df = df.fillna(value=0)
# NaN을 value로 채움

df.isnull()
# 각 요소가 NaN인지 아닌지를 True, False로 반환

df.drop_duplicates(['k1','k2'])
# 특정 column을 기준으로 중복 제거

df.replace(5,-100)
# df 내 특정 값들을 지정한 값으로 변경
```





# 일반적으로 사용되어지는 데이터 표현 방식(3가지)

## 1. CSV(Comma Seperated Values)

장점 : 많은 데이터를 표현하기에 적합, 데이터 사이즈를 작게 할 수 있다.

단점 : 데이터의 구성을 알기 어렵고, 구조적 데이터 표현이 힘들어요. 또한, 사용이 힘들고, 데이터처리를 위해서 따로 프로그램을 만들어야 한다. 데이터가 변경됬을때 프로그램도 같이 변경해야기 때문에 유지보수문제가 발생

즉, 데이터의 크기가 무지막지하고 데이터의 형태가 잘 변하지 않는 경우는 CSV가 가장 알맞은 형태



## 2. XML(eXtended Markup Language)

장점 : 데이터의 구성을 알기 쉽고, 사용하기 편하다. 또한, 프로그램적 유지보수가 쉽다.

단점 : 부가적인 데이터가 많다.

예) <person><name>홍길동</name><age>20</age>

​	  <person><name>김길동</name><age>30</age>



## 3. JSON(JavaScript Object Notation)

현재 일반적인 데이터 표현방식으로 자바스크립트 객체표현방식을 이용해서 데이터를 표현한다. JSON은 데이터 표현방식이지 특정 프로그래밍 언어와는 상관이 없다.

장점 : XML의 장점에 더해 XML보다 용량이 작다.

단점 : CSV에 비해서는 부가적인 데이터가 많다.

예) { name : '홍길동', age : 20, address : '서울' }