from django.db import models


# bbs_post라는 이름의 table로 DB에 생성
# class의 속성이 table의 column이 됨
class Post(models.Model):
    author = models.CharField('작성자', max_length=20)
    contents = models.CharField('글 내용', max_length=100)

    def __str__(self):
        return self.contents
