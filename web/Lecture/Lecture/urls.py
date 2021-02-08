from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import TemplateView
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', include('users.urls')),
    path('movie/', include('boxoffice.urls')),
    # path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('', views.home, name='home')
]
