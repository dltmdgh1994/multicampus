function my_search(){
    let keyword = '건축학개론'

    $.ajax({
        url: 'https://dapi.kakao.com/v2/search/image', // 호출할 서버쪽 프로그램의 URL
        type: 'GET',   // 서버쪽 프로그램에 대한 request 방식
        dataType: 'json',  // 결과의 datatype
        data : {
            query : keyword
        },
        headers : {
            Authorization : 'KakaoAK {REST API키}'
        },
        success : function (){
            alert("호출 성공!");
        },
        error : function (){
            alert("호출 실패!");
        }
    })
}