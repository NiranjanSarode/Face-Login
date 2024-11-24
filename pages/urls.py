from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('',views.start_page,name='start-page'),
    path('video_feed_1/',views.video_feed_1,name='video-feed-1'),
    path('video_feed_2/',views.video_feed_2,name='video-feed-2'),
    path('register/',views.register,name='register'),
    path('login/',views.loginuser,name='login'),
    path('logout/',views.logoutuser,name='logout'),
    path('start_camera/',views.start_camera_1,name='sc-1'),
    path('capture_image/', views.capture_image, name='capture_image'),
    path('reg_face_count/',views.register_count_stream,name='reg_face_count'),
    path('login_react_1/',views.login_reactive_1,name='login_react_1'),
    path('login_react_2/',views.login_reactive_2,name='login_react_2'),
    path('login_react_3/',views.login_reactive_3,name='login_react_3'),
    path('homepage/',views.homepage,name='homepage'),
]
