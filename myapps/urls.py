from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.query_view, name='query_view'),  # Set query_view as the main page
    path('upload_csv/', views.upload_csv_and_create_table, name='upload_csv'),
    # path('edit_schema/<str:table_name>/', views.edit_schema, name='edit_schema'),
    path('save_edited_schema/', views.save_edited_schema, name='save_edited_schema'),
    path('clear_history/', views.clear_history, name='clear_history'),
    path('download_history_xml/', views.download_history_xml, name='download_history_xml'),
    # path('show_table_data/<str:table_name>/', views.show_table_data, name='show_table_data')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
