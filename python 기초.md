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

   list, tuple, ...

   * sort()나 reverse() 같은 함수는 리턴을 하지 않고 원본을 건든다!

   * a = (1,2,3)
     b = (4,5,6)
     print(a + b)   # (1, 2, 3, 4, 5, 6)

   * a = a * 2

     print(a)   # (1, 2, 3, 1, 2, 3)

   * list는 안의 값을 변경할 수 있으나 tuple은 변경할 수 없다.

3. text sequence(문자열)

4. mapping

5. set

6. bool

   