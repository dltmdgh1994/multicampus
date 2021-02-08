from django.urls import path, include
from . import views

app_name = 'users'

urlpatterns = [
    path('login/', views.login, name='login'),
    path('signup/', views.signup, name='signup'),
    path('signupProcess/', views.signup_process, name='signup_process'),
    path('loginProcess/', views.login_process, name='login_process'),
    path('logout/', views.logout, name='logout')
]
