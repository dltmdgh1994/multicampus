function hello(){
    alert("버튼이 클릭되었습니다!")
    user_key = $ ('#userKey').val()
    date = $ ('#date').val()
    open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'
    url = open_api + '?key=' + user_key + '&targetDt=' + date
    location.href = url
}