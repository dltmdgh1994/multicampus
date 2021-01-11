## 1. 가상환경 실행

1. pip 최신버전 upgrade

   python -m pip install --upgrade pip

2. conda 가상환경 생성

   conda create -n 이름 python=3.7 openssl

3. 가상환경 전환

   conda activate 이름

4. 가상환경에 package 설치

   conda install nb_conda 등등 특정 가상환경에서 사용할 패키지 설치
   
   

*  가상환경이 필요한 이유

  여러 프로젝트를 동시에 진행하다 보면 여러 라이브러리, 패키지를 사용하게 되는데 각 라이브러리가 충돌하는 경우가 잦다. 따라서, 이를 방지하기 위해 프로젝트마다 독립적인 환경을 만들어주기 위해 프로젝트 단위로 가상환경을 구성한다.



## 2. jupyter-notebook 홈 디렉토리 변경

1. anaconda 프롬프트에서 jupyter notebook --generate-config를 통해 환경설정 파일 생성

2. `.jupyter`안에 있는 jupyter_notebook_config.py에서#c.NotebookApp.notebook_dir = '' 의 주석을 없애고, ''안에 원하는 경로 입력(단, `\`말고 / 로 경로 설정)

3. jupyter-notebook 재실행

   

## 3. Python data type

1. Numeric(숫자형)

   정수(int), 실수(float), 복소수(complex)

   * 지수표현 : 3 `**` 4 = 81

2. sequence

   list, tuple, range...

   * sort()나 reverse() 같은 함수는 리턴을 하지 않고 원본을 건든다!

   * a = (1,2,3)
     b = (4,5,6)
     print(a + b)   # (1, 2, 3, 4, 5, 6)

   * a = a * 2

     print(a)   # (1, 2, 3, 1, 2, 3)

   * list는 안의 값을 변경할 수 있으나 tuple은 변경할 수 없다.

   * range(시작, 끝, 증가치)

     range(1, 11, 2) # 1, 3 ,5 ,7 ,9

     range(11, 1, -2) # 11, 9, 7, 5, 3 

3. text sequence(문자열)

   str

   * 파이썬은 " "와 ' '을 구분하지 않는다.

   * a = 'python', b = ' basic'

     print(a + b) # 'python basic'

     print(a*2) #'pythonpython'

4. mapping

   dictionary # key와 value의 쌍으로 데이터를 표현

   * a = { 'name' : '홍길동', 'age' : 30, 'address' : '인천' }
   * a.keys()를 통해 키, a.values()를 통해 값, a.items()를 통해 쌍을 추출

5. set

   중복을 배제, 순서가 없는 자료구조

   * a = { 1, 2, 3, 2, 3, 1, 2, 3 } # {1, 2, 3}

   * a = set([1,2,3,4,5]), b = set([4,5,6,7,8])

     a & b => 교집합,  a | b => 합집합, a - b => 차집합

6. bool

   * 사용 가능한 연산자 => and, or, not

   * 빈 문자열, 리스트, Tuple, dict, 숫자 0, None은 False로 간주
   * 파이썬에서 Null은 None이다.

   

## 4. python list comprehension

   리스트를 생성할 때 반복문을 조건문을 이용해서 생성

   ```python
   a = [1,2,3,4,5,6,7]
   
   list1 = [tmp * 2 for tmp in a if tmp % 2 == 0]
   
   print(list1) #[4, 8, 12]
   ```

   

   

  ## 5. Call-by-value, Call-by-reference

```python
# 넘겨준 인자값이 변경되지 않는경우 : call-by-value => immutable (불변의)
# 넘겨준 인자값이 변경되는 경우 : call-by-reference => mutable (가변의)

def my_func(tmp_value, tmp_list):
    tmp_value = tmp_value + 100
    tmp_list.append(100)
    
data_x = 10
data_list = [10,20]

my_func(data_x,data_list)

print('data_x : {}'.format(data_x))   # 10   => immutable (숫자, 문자열, tuple)  ### data_x : 10
print('data_list : {}'.format(data_list)) # [10,20,100]   => mutable (list,dict)   ### data_list : [10, 20, 100]
```



## 6. Class

```python
class Student(object):
    
    scholarship_rate = 3.0  # class variable 
    
    # initializer(생성자-constructor)
    def __init__(self, name, dept, num, grade):
        # 속성을 __init__ 안에서 명시를 해요!
        print('객체가 생성됩니다!!')
        self.name = name   # instance variable
        self.dept = dept
        self.num = num
        self.grade = grade
    
    # instance method
    def get_stu_info(self):
        return '이름 : {}, 학과 : {}'.format(self.name,self.dept)
    
    # class method를 만들려면 특수한 데코레이터를 이용해야 해요!
    # class method는 인스턴스가 공유하는 class variable을 생성, 변경, 참조하		기 위해서 사용되요!
    @classmethod
    def change_scholaship_rate(cls,rate):
        cls.scholarship_rate = rate
        
    # static method를 만들려면 특수한 데코레이터를 이용해야 해요!
    @staticmethod
    def print_hello():
        print('Hello')
        
    # __init__(), __str__(), __del__(), __lt__(), ...
	# 이 magic function의 특징은 일반적으로 우리가 직접 호출하지 않아요!
	# 특정 상황이 되면 자동적으로(내부적으로) 호출되요!
    
    def __del__(self):   # instance가 메모리에서 삭제될 때 호출
        print('객체가 삭제되요!!')
        # 객체가 삭제될 때 이 객체가 사용한 resource를 해제
        
    def __str__(self):
        return '이름은 : {}, 학과는 : {}'.format(self.name, self.dept)
    
    def __gt__(self,other): ## >
        if self.grade > other.grade:
            return True
        else:
            return False
        
    def __lt__(self,other): ## <
        if self.grade < other.grade:
            return True
        else:
            return False 
        
stu1 = Student('강감찬','경영학과','20201120',3.4)
print(stu1.get_stu_info()) ##이름 : 강감찬, 학과 : 경영학과
```



## 7. 상속

```python
# 상위 클래스(super class, parent class, base class)
class Unit(object):
    def __init__(self,damage, life):
        self.utype = self.__class__.__name__ 
        self.damage = damage
        self.life = life
        
# 하위 class(sub class, child class)
class Marine(Unit):
    def __init__(self,damage,life,offense_upgrade):
        super(Marine, self).__init__(damage,life)
        self.offense_upgrade = offense_upgrade

marine_1 = Marine(300,400,2)
print(marine_1.damage)
print(marine_1.utype)
print(marine_1.offense_upgrade)
```



## 8. Module

```python
# module을 이용하는 이유(파일을 나누어서 작성하는 이유)
# 코드의 재사용성을 높이고 관리를 쉽게 하기 위함.

# import : module을 불러들이는 키워드.
#          파일을 객체화 시켜서 우리 코드가 사용하는 메모리에 로드.

import module1 as m1   
from module1 import my_pi
from network.my_sub_folder import my_network_module

```



## 9. 예외처리

```python
	try:
        print('일단 실행') 
        
    except Exception as err:
        print('실행시 문제가 발생했어요!!')
    
    else:
        print('실행시 문제가 없어요!!')
        
    finally:
        print('만약 finally가 존재하면 무조건 실행되요!!')
        
```



## 10. 파일 입출력

```python
my_file = open('mpg.txt','r')

# 파일안에 있는 모든 내용(line)을 화면에 출력할 꺼예요!
# ''  공백문자열은 False로 간주
while True:
    line = my_file.readline()
    print(line)
    if not line:
        break

my_file.close()   # 반드시 사용한 resouce는 적절하게 해제처리를 해 줘야 해요!  
```



## 11. Numpy

```python
import numpy as np

# numpy의 ndarray
b = np.array([1,2,3,4])
print(b)       # [1 2 3 4]
print(type(b)) # <class 'numpy.ndarray'>
print(b.dtype) # int32
# ndarray는 모든 원소가 같은 데이터 타입을 가져야 해요!
# list는 모든 원소가 같은 데이터 타입을 가지지 않아도 상관없어요!
print(type(b[0])) # <class 'numpy.int32'>

# 다차원 ndarray에 대해서 알아보아요!
my_list = [[1,2,3], [4,5,6]]
print(my_list)   # [[1, 2, 3], [4, 5, 6]]

my_array = np.array([[1,2,3], [4,5,6]], dtype=np.float64)
print(my_array)  # [[1 2 3]
                 #  [4 5 6]]
print(my_array[1,1])  # 5
```



### 1. ndarray의 속성

```python
import numpy as np
# ndim : 차원의 수를 표현
# shape : 차원과 요소개수를 tuple로 표현
# size : 배열의 요소 전체의 개수

my_list = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
arr = np.array(my_list)
print(arr)
print(arr.ndim)  # 3
print(arr.shape) # (2,2,3)

my_list = [[1,2,3], [4,5,6]]
arr = np.array(my_list)
print('shape : {}'.format(arr.shape)) # (2,3)
print('크기(len) : {}'.format(len(arr))) # 2 가장 상위의 차원의 수
print('크기(size) : {}'.format(arr.size)) # 6

# shape을 변경할때는 이렇게 직접적으로 shape속성을 변경하지 않아요!
# reshape를 사용


```

### 2. ndarray의 메서드

```python
import numpy as np

# astype() ndarray의 data type을 변경
arr = np.array([1.2,2.3,3.5,4.1,5.7])
arr = arr.astype(np.int32)
print(arr) ## [1 2 3 4 5]

# 특정 형태의 ndarray를 만들어서 내용을 0으로 채움
# 기본 데이터 타입은 np.float64 
arr = np.zeros((3,4), dtype=np.int32)

# 특정 형태의 ndarray를 만들어서 내용을 1으로 채움
arr = np.ones((3,4), dtype=np.float64)

# 초기화를 하지 않기 때문에 빠르게 공간만 설정
arr = np.empty((3,4))   

# 특정 숫자로 내용을 채움
arr = np.full((3,4), 7, dtype=np.float64)

# numpy arange()
arr = np.arange(0,10,1) # [ 0  1  2  3  4  5  6  7  8  9 ]
arr = np.arange(10) # 위와 동일


# random기반의 생성방법(방법이 5가지 정도)
# 1. np.random.normal() : 정규분포에서 실수형태의 난수를 추출
my_mean = 50
my_std = 2
arr = np.random.normal(my_mean,my_std,(10000,))

# 2. np.random.rand() : 0이상 1미만의 실수를 난수로 추출
#                       균등분포로 난수를 추출
# np.random.rand(d0, d1, d2, d3, ...)   dn : 각 차원의 요소 갯수    
arr = np.random.rand(10000)

# 3. np.random.randn() : 표준 정규분포에서 실수형태로 난수를 추출
# np.random.randn(d0, d1, d2, d3, ...)
arr = np.random.randn(10000)

# 4. np.random.randint(low,high,shape) : 균등분포로 정수 표본을 추출
arr = np.random.randint(10,100,(10000,)) # 10 ~ 100사이 정수

# 5. np.random.random() : 0이상 1미만의 실수를 난수로 추출
#                         균등분포로 난수를 추출
#    np.random.rand(d0, d1, d2, d3, ...)     
arr = np.random.random((10000,))


# 시드 설정
# 실행할 때 마다 같은 난수가 추출되도록 설정(난수의 재현)
np.random.seed(3)  # 정수만 사용되고 음수는 사용할 수 없어요!


# reshape() => ndarray의 형태를 조절
# reshape()함수는 새로운 ndarray를 만들지 않고 view가 생성
# 메모리를 아끼기 위함
arr = np.arange(12)  # 12개의 요소를 가지는 1차원의 ndarray
arr1 = arr.reshape(3,4)  # 3행 4열의 2차원의 ndarray로 바꿀 수 있어요!
arr1 = arr.reshape(2,3,-1)  # -1은 특별한 의미를 가져요!
							# (2, 3, 2) 로 알아서 생성

# view가 아니라 새로운 배열을 생성하고자 할 때는 copy() 이용
arr1 = arr.reshape(3,4).copy()

# ravel() => ndarray가 가지고있는 모든 요소를 포함하는 1차원의 ndarray로 변경
# ravel()함수는 View를 리턴
arr1 = arr.ravel()

# resize() => 결과를 리턴하지 않고 원본을 바꿔요!
arr.resize(2, 6)
arr1 = np.resize(arr,(1,6))  # 원본은 불변, 복사복이 만들어져요!
```

