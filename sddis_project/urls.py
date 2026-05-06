from django.urls import path
from dashboard import views
urlpatterns = [
    path('', views.index, name='index'),
    path('api/forecast/', views.forecast_api, name='forecast_api'),
    path('api/chatbot/',  views.chatbot_api,  name='chatbot_api'),
]
