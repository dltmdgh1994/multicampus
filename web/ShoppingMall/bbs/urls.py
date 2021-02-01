from django.urls import path
from . import views

app_name = 'bbs'

urlpatterns = [
    path('list/', views.p_list, name='p_list'),
    path('create/', views.p_create, name='p_create'),
    path('<int:post_id>/delete/', views.p_delete, name='p_delete'),
    path('<int:post_id>/update/', views.p_update, name='p_update')
]
