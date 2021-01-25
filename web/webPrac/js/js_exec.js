function fuc(){
    alert("버튼이 클릭되었습니다!")
    $('*').css('color','red');
    console.log($('#apple').text())
    // .은 클래스 속성을 지칭
    console.log($('ul >.myList').text())


}