# 1. 프로젝트 및 애플리케이션 생성

```bash
# 프로젝트 생성
django-admain startproject mysite

# polls라는 애플리케이션 생성(투표기능)
# manage.py가 있는 곳으로 cd한 후
python manage.py startapp polls
```



# 2. Settings.py 변경

```bash
# 1. ALLOWED_HOSTS 지정(서버의 IP나 도메인을 지정)
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# 2. INSTALLED_APPS 에 'polls.apps.PollsConfig' 추가

# 3. 사용할 DB 엔진 설정 (default : SQLite3)

# 4. 타임존 지정
TIME_ZONE = 'Asia/Seoul'
```



# 3. migrate 및 runserver

```bash
# 파이참 터미널에서 명령어 수행
# default db 생성 (migrate는 DB에 변경사항이 있을 경우 이를 반영해주는 명령어)
python manage.py migrate

# 서버 가동
python manage.py runserver
# 서버 가동 후 http://localhost:8000/로 접속
# 관리자 페이지 : http://localhost:8000/admin

#서버 종료
#컨트롤+c 입력

# 관리자 페이지 로그인을 위해 슈퍼계정 생성
# 대상 프로젝트 폴더로 cd한 후
python manage.py createsuperuser
```



# 4. 애플리케이션 개발

![프로젝트 폴더 구조](md-images/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%ED%8F%B4%EB%8D%94%20%EA%B5%AC%EC%A1%B0.PNG)

## 1. Model 코딩

1. 테이블 정의 (polls 애플리케이션은 Question과 Choice 2개의 테이블이 필요)

   ```bash
   # models.py에 구현
   from django.db import models
   
   
   # Create your models here.
   class Question(models.Model):
       # 이렇게 정의되는 class가 DB의 Table과 mapping
       # Table의 column를 속성으로 표현
       question_text = models.CharField(max_length=200)
       pub_date = models.DateTimeField('date published')
   
       def __str__(self):
           return self.question_text
   
   
   class Choice(models.Model):
       choice_text = models.CharField(max_length=200)
       votes = models.IntegerField(default=0)
       question = models.ForeignKey(Question, on_delete=models.CASCADE)
   
       def __str__(self):
           return self.choice_text
   ```

   

2. Admin에 테이블 반영

   ```bash
   # admin.py에 구현
   from django.contrib import admin
   from polls.models import Question, Choice
   
   
   # Register your models here.
   admin.site.register(Question)
   admin.site.register(Choice)
   ```



3. DB 변경사항 반영

   ```bash
   # 파이참 터미널에서 명령어 수행
   python manage.py makemigrations
   python manage.py migrate
   ```



## 2. View 및 Templete 코딩

![APP 구조](md-images/APP%20%EA%B5%AC%EC%A1%B0.PNG)

1. 