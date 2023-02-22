from django.urls import path
from . import views

urlpatterns = [
    path('', views.stream_video,  name='stream_video'),
    # path('video_feed/', views.video_feed, name='video_feed'),
]
