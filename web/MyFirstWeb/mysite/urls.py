from django.contrib import admin
from django.urls import path, include
from polls import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # path('polls/', views.index, name='index')
    # 애플리케이션 단위로 관리하기 위해 include 사용
    path('polls/', include('polls.urls'))
]
