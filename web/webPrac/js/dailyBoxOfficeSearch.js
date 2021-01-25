
function load_data(){
    let user_date = $('#userInputDate').val();
    let user_key = '7722ddf37f0243d6b83affbdab6c38ce';
    let open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'

    // refresh가 일어나서 화면을 갱신함
    // let url = open_api + '?key=' + user_key + '&targetDt=' + user_date;
    // location.href = url;

    //이 문제를 해결하기 위해 AJAX를 사용
    $.ajax({
        url : open_api, // 호출할 서버쪽 프로그램의 URL
        type : 'GET',   // 서버쪽 프로그램에 대한 request 방식
        dataType : 'json',  // 결과의 datatype
        data : {  // 서버에 보내줘야할 데이터
            key : user_key,
            targetDt : user_date
        },
        success : function (){ // 성공 시 호출
            alert("호출 성공!");
        },
        error : function (){ // 실패 시 호출
            alert("호출 실패!");
        }
    });


}