# 여기서는 ModelForm class를 정의
# ModelForm이 자동으로 Form field(HTML tag)를 생성
# => Form 처리를 간단하게 해줌
from django import forms
from bbs.models import Post


class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['author', 'contents']
