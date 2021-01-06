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



## 6. Class 선언

```python
class Student(object):
    # initializer(생성자-constructor)
    def __init__(self, name, dept, num, grade):
        # 속성을 __init__ 안에서 명시를 해요!
        self.name = name   # 전달된 값으로 이름속성을 설정
        self.dept = dept
        self.num = num
        self.grade = grade
    
    # 아래의 method는 객체가 가지고 있는 학생의 정보를 문자열로
    # 리턴하는 역할을 수행하는 method
    def get_stu_info(self):
        return '이름 : {}, 학과 : {}'.format(self.name,self.dept)
        
stu1 = Student('강감찬','경영학과','20201120',3.4)
print(stu1.get_stu_info()) ##이름 : 강감찬, 학과 : 경영학과
```

