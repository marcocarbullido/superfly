from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.contrib.auth.models import User


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'accounts/register.html', {'form': form})


def home(request):
@login_required
def home(request):
    return render(request, 'accounts/home.html')
@login_required
def admin_panel(request):
    if not request.user.is_superuser:
        return HttpResponseForbidden("You do not have permission to view this page.")
    users = User.objects.all().order_by('username')
    return render(request, 'accounts/admin.html', {'users': users})