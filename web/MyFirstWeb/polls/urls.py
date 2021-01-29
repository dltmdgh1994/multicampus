from django.urls import path, include
from . import views

# index 변수를 공유하기 때문에 namespace를 지정
app_name = 'polls'

urlpatterns = [
    path('', views.index, name='index'),
    # 변하는 값을 넣기 위해 <int:>를 사용
    path('<int:question_id>/', views.detail, name='detail'),    # polls:detail
    path('<int:question_id>/vote/', views.vote, name='vote'),   # polls:vote
    path('<int:question_id>/results/', views.results, name='results')   # polls:results
]
