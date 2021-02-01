function new_post(){
    location.href = '/bbs/create'
    // bbs/crete => 상대경로
    // /bbs/create => 절대경로
}

function delete_post(post_id){
    location.href = '/bbs/'+post_id+'/delete'
}

function update_post(post_id){
    location.href = '/bbs/'+post_id+'/update'
}