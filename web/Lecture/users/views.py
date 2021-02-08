from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import auth


# Create your views here.
def login(request):
    # DB 처리가 있으면 model을 이용해서 데이터를 가져옴
    # 로직 처리가 필요하면 로직 처리를 함
    # templates를 이용해서 결과 HTML를 리턴
    return render(request, 'users/login.html', {
        'page_title': 'User Login',
        'user_data': '소리없는 아우성!!'
    })


def logout(request):
    # 로그아웃 처리를 진행(session 정보를 만료 처리)
    auth.logout(request)
    return redirect('home')


def signup(request):
    return render(request, 'users/signup.html', {
        'page_title': 'User Sign Up'
    })


def signup_process(request):
    user_id = request.POST['inputId']
    u_pass1 = request.POST['inputPassword1']
    u_pass2 = request.POST['inputPassword2']

    user_list = User.objects.all()
    if user_list.filter(username=user_id).exists():
        return render(request, 'users/signup.html', {
            'err_msg': '존재하는 ID입니다'
        })
    elif u_pass1 == u_pass2:
        User.objects.create_user(username=user_id, password=u_pass1)
        redirect('home')
    else:
        return render(request, 'users/signup.html', {
            'err_msg': '비밀번호가 다릅니다'
        })


def login_process(request):
    u_id = request.POST['inputId']
    u_pw = request.POST['inputPassword']

    user = auth.authenticate(request, username=u_id, password=u_pw)
    if user is not None:
        # 로그인 처리를 진행(session 처리 진행)
        auth.login(request, user)

        user_dict = {
            'u_id': user.id,
            'u_name': user.username
        }
        request.session['loginObj'] = user_dict
        return redirect('home')
    else:
        return render(request, 'users/login.html', {
            'err_msg': '로그인 실패'
        })
