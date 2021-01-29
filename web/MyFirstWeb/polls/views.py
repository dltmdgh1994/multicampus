from django.shortcuts import render, get_object_or_404
from polls.models import Question, Choice
from django.http import HttpResponseRedirect
from django.urls import reverse


# DB에서 설문목록을 가져옴
def index(request):
    # 오름차순
    # 내림차순은 '-pub_date'
    question_list = Question.objects.all().order_by('pub_date')[:5]

    # 데이터 전달용 dic
    context = {'q_list': question_list}

    return render(request, 'polls/index.html', context)


# 숫자가 question_id로, question_id는 설문에 대한 PK
# 투표에 진입 후 선택지를 가져옴
def detail(request, question_id):
    # object를 불러오고, 실패하면 404페이지 띄움
    question = get_object_or_404(Question, pk=question_id)

    context = {'selected_question': question}

    return render(request, 'polls/detail.html', context)


# 투표한 경우 DB 내 표 수를 변경
def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['my_choice'])
    except(KeyError, Choice.DoesNotExist):
        # 선택지를 선택하지 않아서 오류가 발생하는 경우
        return render(request, 'polls/detail.html',
                      {'selected_question': question,
                       'error_message': 'Select Something!'})
    else:
        selected_choice.votes += 1
        selected_choice.save()

        # reverse() => urls.py(URLConf)에 있는 name을 이용해서 url형식으로 반환
        return HttpResponseRedirect(reverse('polls:results',
                                            # 튜플을 표현하기 위해 요소 한개는 ,를 추가
                                            args=(question.id,)))


# 투표 결과를 보여줌
def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'polls/results.html', {'question': question})
