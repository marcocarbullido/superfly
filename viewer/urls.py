from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('mask/<int:index>/', views.mask_overlay, name='mask_overlay'),
    # returns only colored mask overlay (transparent background)
    path('mask-only/<int:index>/', views.mask_only, name='mask_only'),
    # returns list of mask ids present in the given frame
    path('mask-ids/<int:index>/', views.mask_ids, name='mask_ids'),
    # serve base image as PNG
    path('image/<int:index>/', views.base_image, name='base_image'),
    # serve low-resolution base image for progressive loading
    path('image-lowres/<int:index>/', views.base_image_lowres, name='base_image_lowres'),
    # endpoint to edit a mask (brush strokes)
    path('edit-mask/', views.edit_mask, name='edit_mask'),
    # endpoint to merge selected objects across frames
    path('merge-objects/', views.merge_objects, name='merge_objects'),
    # dynamic crop endpoints for progressive view
    path('crop-image/<int:index>/', views.crop_image, name='crop_image'),
    path('crop-mask/<int:index>/', views.crop_mask, name='crop_mask'),
]