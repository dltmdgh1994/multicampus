
function my_func() {
    // 버튼을 누르면 호출되요!!
    // jQuery 사용법
    // 1. selector부터 알아보아요!!
    // selector는 HTML엘리먼트를 지칭하는 특수한 표기법을 의미
    // jQuery는 $로 시작해요!
    // $(selector).method()
    // 1. 전체 선택자 : *
    // $('*').css('color','red');
    // 2. 태그 선택자 : 태그명을 가지고 선택
    // $('span').remove()
    // $('li').css('background-color','yellow')
    // 3. 아이디 선택자 : ID속성을 이용해서 선택
    // $('#inchon').text('소리없는 아우성!!')
    // 4. 클래스 선택자 : class속성을 이용해서 선택
    // $('.region').css('color','blue')
    // 5. 구조 선택자 : 부모, 자식, 형제 관계를 이용해서 선택

    // $('div').css('color','red')
    // $('div').css('background-color','yellow')
    //$('div').addClass('myStyle')
    //$('input[type=button]:first').removeAttr('disabled')
    // $('div.myStyle').remove()
    // $('div.myStyle').empty()  // 자신은 삭제하지 말고 자신의 후손을 모두 삭제
    // 그럼 없는 element를 만들려면 어떻게 해야 하나요??

    // <div>소리없는 아우성</div>
    // let my_div = $('<div></div>').text('소리없는 아우성')
    // 위와 같은 방법으로 없는 element를 새롭게 생성할 수 있어요|!!
    // let my_img = $('<img />').attr('src','img/car.jpg')
    // <img src=img/car.jpg>
    // 이렇게 새로운 element를 만들었으면 내가 원하는 위치에 가져다 붙여야 해요!
    // 4종류의 함수를 이용해서 element를 원하는 위치에 가져다 붙일 수 있어요!
    // 1. append() : 자식으로 붙이고 맨 마지막 자식으로 붙여요!
    // 2. prepend() : 자식으로 붙이고 맨 처음 자식으로 붙여요!
    // 3. after() : 형제로 붙이고 바로 다음 형제로 붙여요!
    // 4. before() : 형제로 붙이고 바로 이전 형제로 붙여요!

    // 새로운  li를 생성할 꺼예요!
    let my_li = $('<li></li>').text('아이유')          // <li>아이유</li>
    // $('ul').append(my_li)
    // $('ul').prepend(my_li)
    // $('ul > li:eq(1)').after(my_li)
    // $('ul > li:last').before(my_li)
}