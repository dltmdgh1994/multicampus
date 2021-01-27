
function load_data(){
    let user_date = $('#userInputDate').val();
    let user_key = '{영화진흥원 REST_KEY}';
    let open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'

    // refresh가 일어나서 화면을 갱신함
    // let url = open_api + '?key=' + user_key + '&targetDt=' + user_date;
    // location.href = url;

    //이 문제를 해결하기 위해 AJAX를 사용
    $.ajax({
        url : open_api, // 호출할 서버쪽 프로그램의 URL
        type : 'GET',   // 서버쪽 프로그램에 대한 request 방식
        dataType : 'json',  // 결과의 datatype
        data : {   // 서버에 보내줘야할 데이터
            key : user_key,
            targetDt : user_date
        },
        success : function (result){ // 성공 시 호출
            $('#my_tbody').empty()
            // alert("호출 성공!");
            let movie_list = result['boxOfficeResult']['dailyBoxOfficeList']
            for(let i = 0; i < movie_list.length; i++){
                let m_name = movie_list[i].movieNm
                let m_rank = movie_list[i].rank
                let m_sales = movie_list[i].salesAcc
                let m_audiAcc = movie_list[i].audiAcc

                // 데이터 생성 후 HTML element를 생성
                let tr = $('<tr></tr>')
                let name_td = $('<td></td>').text(m_name)
                let rank_td = $('<td></td>').text(m_rank)
                let sales_td = $('<td></td>').text(m_sales)
                let audiAcc_td = $('<td></td>').text(m_audiAcc)
                // 1. 인덱스를 통해 별개의 id 부여
                //var img_id = "poster_img" + i.toString()
                //let img_td = $('<td></td>').attr('id',img_id)
                
                // 2. 통합된 id 부여
                let img_td = $('<td></td>').attr('id','img_id')

                let poster_td = $('<td></td>')
                let poster_btn = $('<Input/>').attr('type','button')
                    .attr('value','포스터 보기')

                poster_btn.on('click', function () {
                    let p = $('<img></img>')
                    $.ajax({
                        url: 'https://dapi.kakao.com/v2/search/image', // 호출할 서버쪽 프로그램의 URL
                        type: 'GET',   // 서버쪽 프로그램에 대한 request 방식
                        dataType: 'json',  // 결과의 datatype
                        data : {
                            query : m_name
                        },
                        headers : {
                            Authorization : 'KakaoAK {카카오 REST_KEY}'
                        },
                        success : function (r){
                            //alert("호출 성공!");
                            let url = r['documents'][0]['image_url']
                            p.attr('src',url)
                            // 1. 별개의 id를 통해 접근
                            //var poster_img_id = '#poster_img'+i.toString()
                            //$(poster_img_id).append(p)
                        },
                        error : function (){
                            alert("호출 실패!");
                        }
                    })
                    // 2. 통합된 id를 통해 접근
                    $(this).parent().parent().children('#img_id').append(p)
                })

                poster_td.append(poster_btn)

                let delete_td = $('<td></td>')
                let delete_btn = $('<Input/>').attr('type','button')
                    .attr('value','삭제')

                delete_btn.on('click', function() {
                    $(this).parent().parent().remove()
                })

                delete_td.append(delete_btn)
                tr.append(name_td)
                tr.append(rank_td)
                tr.append(sales_td)
                tr.append(audiAcc_td)
                tr.append(img_td)
                tr.append(poster_td)
                tr.append(delete_td)

                $('#my_tbody').append(tr)
            }
        },
        error : function (){ // 실패 시 호출
            alert("호출 실패!");
        }
    });
}
