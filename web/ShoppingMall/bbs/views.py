from django.shortcuts import render, redirect, get_object_or_404
from bbs.models import Post
from bbs.forms import PostForm


def p_list(request):
    # DB의 모든 글의 내용을 들고 옴
    posts = Post.objects.all().order_by('-id')
    return render(request, 'bbs/list.html', {'posts': posts})


def p_create(request):
    # GET 방식
    if request.method == 'GET':
        # 빈 입력 form을 출력
        post_form = PostForm()
        return render(request, 'bbs/create.html',
                      {'post_form': post_form})

    # POST 방식
    if request.method == 'POST':
        # DB에 저장
        post_form = PostForm(request.POST)

        if post_form.is_valid():
            post_form.save()
            return redirect('bbs:p_list')


def p_delete(request, post_id):
    post = get_object_or_404(Post, pk=post_id)
    post.delete()
    return redirect('bbs:p_list')


def p_update(request, post_id):
    post = get_object_or_404(Post, pk=post_id)
    if request.method == 'GET':
        # 빈 입력 form을 출력
        post_form = PostForm(instance=post)
        return render(request, 'bbs/update.html',
                      {'post_form': post_form})

    if request.method == 'POST':
        # 수정
        post_form = PostForm(request.POST)

        if post_form.is_valid():
            post.author = post_form.cleaned_data['author']
            post.contents = post_form.cleaned_data['contents']
            post.save()
            return redirect('bbs:p_list')
