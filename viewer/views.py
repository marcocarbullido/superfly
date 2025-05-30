import os
import glob
from io import BytesIO
import numpy as np
from PIL import Image, UnidentifiedImageError
import colorsys
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse, Http404, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from django.views.decorators.http import require_GET
from functools import lru_cache

# Cache loaded images in memory to speed up cropping operations
@lru_cache(maxsize=10)
def get_base_image(index):
    """
    Load and cache the base TIFF image for the given frame index.
    Returns a PIL Image in RGB mode.
    """
    path = os.path.join(settings.MEDIA_ROOT, 'img', f'img_6_10_{index}.tif')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base image not found: {path}")
    img = Image.open(path)
    # ensure image is fully loaded and decoupled from file
    img = img.convert('RGBA').copy()
    img.load()
    return img

@lru_cache(maxsize=10)
def get_mask_only_image(index):
    """
    Load and cache the machine mask PNG for the given frame index,
    returning a PIL Image in RGBA mode with background transparent.
    """
    path = os.path.join(settings.MEDIA_ROOT, 'masks_machine', f'img_6_10_{index}.png')
    if not os.path.exists(path):
        return None
    m = Image.open(path).convert('RGBA')
    arr = np.array(m)
    # make background (black) transparent
    mask_fg = (arr[..., :3] != 0).any(axis=2)
    arr[..., 3] = (mask_fg * 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGBA')

def index(request):
    image_dir = os.path.join(settings.MEDIA_ROOT, 'img')
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    indices = [int(f.split('_')[-1].split('.')[0]) for f in image_files]
    indices.sort()
    count = len(indices)
    # determine dimensions of first image to constrain zoom-out
    if indices:
        try:
            img0 = get_base_image(indices[0])
            orig_width, orig_height = img0.size
        except Exception:
            orig_width, orig_height = 0, 0
    else:
        orig_width, orig_height = 0, 0
    return render(request, 'index.html', {
        'indices': indices,
        'count': count,
        'media_url': settings.MEDIA_URL,
        'origWidth': orig_width,
        'origHeight': orig_height,
    })

def mask_overlay(request, index):
    """Return base image overlaid with machine-generated mask as PNG."""
    base_path = os.path.join(settings.MEDIA_ROOT, 'img', f'img_6_10_{index}.tif')
    if not os.path.exists(base_path):
        raise Http404("Image not found")
    base_img = Image.open(base_path)
    base_rgba = base_img.convert("RGBA")
    # load machine mask PNG
    mask_path = os.path.join(settings.MEDIA_ROOT, 'masks_machine', f'img_6_10_{index}.png')
    if os.path.exists(mask_path):
        try:
            m = Image.open(mask_path).convert("RGBA")
        except UnidentifiedImageError:
            composite = base_rgba
        else:
            arr = np.array(m)
            # create alpha mask: pixels not black
            mask_fg = (arr[..., :3] != 0).any(axis=2)
            # set desired transparency
            alpha_val = 100
            arr[..., 3] = (mask_fg * alpha_val).astype(np.uint8)
            mask_rgba = Image.fromarray(arr, mode="RGBA")
            composite = Image.alpha_composite(base_rgba, mask_rgba)
    else:
        composite = base_rgba
    buf = BytesIO()
    composite.save(buf, format="PNG")
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/png")
    
def mask_only(request, index):
    """Return only the machine mask overlay on transparent background as PNG."""
    # Load cached mask-only image (RGBA with transparent background)
    out = get_mask_only_image(index)
    if out is None:
        raise Http404("Mask not found")
    buf = BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/png")

def mask_ids(request, index):
    """Return JSON list of machine mask class IDs present in the given frame."""
    # load color mapping for classes
    mapping_path = os.path.join(settings.MEDIA_ROOT, 'obj_class_to_machine_color.json')
    try:
        with open(mapping_path, 'r') as f:
            color_map = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return JsonResponse([], safe=False)
    # invert mapping: color tuple -> id
    color_to_id = {tuple(v): int(k.split('_')[1]) for k, v in color_map.items() if isinstance(v, list) and len(v) == 3}
    mask_path = os.path.join(settings.MEDIA_ROOT, 'masks_machine', f'img_6_10_{index}.png')
    if not os.path.exists(mask_path):
        return JsonResponse([], safe=False)
    try:
        m = Image.open(mask_path).convert("RGB")
    except UnidentifiedImageError:
        return JsonResponse([], safe=False)
    arr = np.array(m)
    # find unique colors excluding background black
    colors = {tuple(col) for col in arr.reshape(-1, 3) if tuple(col) != (0, 0, 0)}
    ids = []
    for col in colors:
        obj_id = color_to_id.get(col)
        if obj_id is not None:
            ids.append(obj_id)
    ids = sorted(set(ids))
    return JsonResponse(ids, safe=False)

def base_image(request, index):
    """Return the base image converted to PNG."""
    base_path = os.path.join(settings.MEDIA_ROOT, 'img', f'img_6_10_{index}.tif')
    if not os.path.exists(base_path):
        raise Http404("Image not found")
    img = Image.open(base_path)
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/png")
    
def base_image_lowres(request, index):
    """
    Return a low-resolution version of the base image as PNG for progressive loading.
    Scales the longest dimension to 256 pixels.
    """
    base_path = os.path.join(settings.MEDIA_ROOT, 'img', f'img_6_10_{index}.tif')
    if not os.path.exists(base_path):
        raise Http404("Image not found")
    try:
        img = Image.open(base_path)
    except UnidentifiedImageError:
        raise Http404("Image not found")
    # create thumbnail with longest side = 256px
    max_dim = 256
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/png")
        
@csrf_exempt
@require_POST
def edit_mask(request):
    """Edit a specific object mask by drawing a filled circle brush stroke."""
    try:
        data = json.loads(request.body)
        frame_idx = int(data.get('index'))
        obj_id = int(data.get('id'))
        x = int(data.get('x'))
        y = int(data.get('y'))
        size = int(data.get('size'))
        action = data.get('action', 'add')  # 'add' or 'subtract'
    except (ValueError, TypeError, json.JSONDecodeError):
        return JsonResponse({'status': 'error', 'message': 'Invalid payload'}, status=400)
    # Determine paths
    masks_dir = os.path.join(settings.MEDIA_ROOT, 'masks_instances', f'img_6_10_{frame_idx}')
    if not os.path.isdir(masks_dir):
        return JsonResponse({'status': 'error', 'message': 'Masks directory not found'}, status=404)
    mask_path = os.path.join(masks_dir, f'object_{obj_id}.tif')
    # Load or initialize mask array
    if os.path.exists(mask_path):
        m = Image.open(mask_path)
        arr = np.array(m)
    else:
        # Create a new blank mask based on base image size
        base_path = os.path.join(settings.MEDIA_ROOT, 'img', f'img_6_10_{frame_idx}.tif')
        if not os.path.exists(base_path):
            return JsonResponse({'status': 'error', 'message': 'Base image not found'}, status=404)
        base_img = Image.open(base_path)
        width, height = base_img.size
        arr = np.zeros((height, width), dtype=np.uint8)
    # Draw filled circle into mask array (add or subtract mode)
    yy, xx = np.ogrid[:arr.shape[0], :arr.shape[1]]
    circle = (yy - y) ** 2 + (xx - x) ** 2 <= size ** 2
    if action == 'subtract':
        arr[circle] = 0
    else:
        arr[circle] = 255
    # Save updated mask
    out_img = Image.fromarray(arr, mode='L')
    out_img.save(mask_path)
    return JsonResponse({'status': 'ok'})
 
@csrf_exempt
@require_POST
def merge_objects(request):
    """Merge selected classes across all frames into a new class ID in the machine masks."""
    try:
        data = json.loads(request.body)
        items = data.get('items', [])
        if not isinstance(items, list) or len(items) < 2:
            return JsonResponse({'status': 'error', 'message': 'Need at least two objects to merge'}, status=400)
    except (json.JSONDecodeError, TypeError):
        return JsonResponse({'status': 'error', 'message': 'Invalid payload'}, status=400)
    # extract unique IDs to merge
    merge_ids = []
    for itm in items:
        try:
            oid = int(itm.get('id'))
        except (TypeError, ValueError):
            continue
        if oid not in merge_ids:
            merge_ids.append(oid)
    # load color mapping
    mapping_path = os.path.join(settings.MEDIA_ROOT, 'obj_class_to_machine_color.json')
    try:
        with open(mapping_path, 'r') as f:
            color_map = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return JsonResponse({'status': 'error', 'message': 'Color map not found'}, status=500)
    # determine new class ID
    existing = []
    for key in color_map.keys():
        parts = key.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            existing.append(int(parts[1]))
    new_id = max(existing) + 1 if existing else 1
    # assign new color via HLS
    h = (new_id * 0.618033988749895) % 1
    r_f, g_f, b_f = colorsys.hls_to_rgb(h, 0.5, 1.0)
    new_color = [int(r_f * 255), int(g_f * 255), int(b_f * 255)]
    color_map[f'object_{new_id}'] = new_color
    # update mapping file
    try:
        with open(mapping_path, 'w') as f:
            json.dump(color_map, f, indent=4)
    except OSError:
        return JsonResponse({'status': 'error', 'message': 'Failed to write color map'}, status=500)
    # update masks in temp/masks_machine
    masks_dir = os.path.join(settings.MEDIA_ROOT, 'masks_machine')
    if not os.path.isdir(masks_dir):
        return JsonResponse({'status': 'error', 'message': 'Masks directory not found'}, status=404)
    # map old IDs to their colors
    id_to_color = {}
    for oid in merge_ids:
        key = f'object_{oid}'
        if key in color_map:
            id_to_color[oid] = tuple(color_map[key])
    # replace pixels in each frame mask
    for fname in os.listdir(masks_dir):
        if not fname.lower().endswith('.png'):
            continue
        path = os.path.join(masks_dir, fname)
        try:
            img = Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            continue
        arr = np.array(img)
        # for each merge ID, replace its color
        for oid, col in id_to_color.items():
            mask = (arr[...,0] == col[0]) & (arr[...,1] == col[1]) & (arr[...,2] == col[2])
            arr[mask] = new_color
        # save updated mask
        out = Image.fromarray(arr, mode='RGB')
        out.save(path)
    # Clear cached mask images so updated masks are served
    get_mask_only_image.cache_clear()
    return JsonResponse({'status': 'ok', 'new_id': new_id})

@require_GET
def crop_image(request, index):
    """
    Return a cropped region of the base image as PNG.
    Query params: x, y, w, h (pixels in original image), ds (downsample factor, >=1).
    The server will crop the region [x:x+w, y:y+h], then resize to (ceil(w/ds), ceil(h/ds)).
    """
    try:
        x = int(request.GET.get('x', 0))
        y = int(request.GET.get('y', 0))
        w = int(request.GET.get('w', 0))
        h = int(request.GET.get('h', 0))
        ds = int(request.GET.get('ds', 1))
        if ds < 1:
            ds = 1
    except (TypeError, ValueError):
        return JsonResponse({'status': 'error', 'message': 'Invalid parameters'}, status=400)
    # Load cached base image or return 404 if missing
    try:
        img = get_base_image(index)
    except FileNotFoundError:
        raise Http404("Image not found")
    # constrain crop within bounds
    img_w, img_h = img.size
    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    x1 = max(0, min(x0 + w, img_w))
    y1 = max(0, min(y0 + h, img_h))
    if x1 <= x0 or y1 <= y0:
        return JsonResponse({'status': 'error', 'message': 'Empty crop region'}, status=400)
    region = img.crop((x0, y0, x1, y1))
    if ds > 1:
        new_w = int((x1 - x0 + ds - 1) // ds)
        new_h = int((y1 - y0 + ds - 1) // ds)
        region = region.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    region.save(buf, format='PNG')
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type='image/png')

@require_GET
def crop_mask(request, index):
    """
    Return a cropped region of the machine mask overlay (transparent PNG).
    Same query params as crop_image.
    """
    try:
        x = int(request.GET.get('x', 0))
        y = int(request.GET.get('y', 0))
        w = int(request.GET.get('w', 0))
        h = int(request.GET.get('h', 0))
        ds = int(request.GET.get('ds', 1))
        if ds < 1:
            ds = 1
    except (TypeError, ValueError):
        return JsonResponse({'status': 'error', 'message': 'Invalid parameters'}, status=400)
    # Load cached mask-only image (RGBA with transparent background)
    mask_img = get_mask_only_image(index)
    if mask_img is None:
        # return empty transparent PNG of requested size
        empty = Image.new('RGBA', (max(w, 1), max(h, 1)), (0, 0, 0, 0))
        buf = BytesIO(); empty.save(buf, format='PNG'); buf.seek(0)
        return HttpResponse(buf.getvalue(), content_type='image/png')
    # constrain crop within image bounds
    img_w, img_h = mask_img.size
    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    x1 = max(0, min(x0 + w, img_w))
    y1 = max(0, min(y0 + h, img_h))
    if x1 <= x0 or y1 <= y0:
        # empty
        empty = Image.new('RGBA', (w if w>0 else 1, h if h>0 else 1), (0,0,0,0))
        buf = BytesIO(); empty.save(buf, format='PNG'); buf.seek(0)
        return HttpResponse(buf.getvalue(), content_type='image/png')
    region = mask_img.crop((x0, y0, x1, y1))
    if ds > 1:
        new_w = int((x1 - x0 + ds - 1) // ds)
        new_h = int((y1 - y0 + ds - 1) // ds)
        region = region.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    region.save(buf, format='PNG')
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type='image/png')