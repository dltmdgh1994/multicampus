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

## dict를 사용하는 방법
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
                
## csv파일을 읽는 방법
df = pd.read_csv('./movies.csv')
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