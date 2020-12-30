# git 기초 명령어

* 분산버전관리시스템(DVCS)

## 0. 로컬 저장소(repository) 설정

```bash
# 초기화
$ git init
Reinitialized existing Git repository in C:/Users/dltmd/OneDrive/바탕 화면/Prac/.git/
```

* `.git` 폴더가 생성되고, 여기에 모든 git과 관련된 정보들이 저장

## 1. add

```bash
$ touch a.txt
$ git status
On branch master
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        "Markdown \353\254\270\353\262\225 \352\270\260\354\264\210.md"
        "git \352\270\260\354\264\210.md"
        md-images/

nothing added to commit but untracked files present (use "git add" to track)

$ git add .
$ git status
On branch master
# 커밋이 변결될 사항
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   "Markdown \353\254\270\353\262\225 \352\270\260\354\264\210.md"
        new file:   "git \352\270\260\354\264\210.md"
        new file:   md-images/md-images.PNG
```

## 2. commit

```bash
$ git commit -m 'Third commit'
[master bbe02e6] Third commit
 3 files changed, 84 insertions(+)
 create mode 100644 "Markdown \353\254\270\353\262\225 \352\270\260\354\264\210.md"
 create mode 100644 "git \352\270\260\354\264\210.md"
 create mode 100644 md-images/md-images.PNG
```

* `commit`은 지금 상태를 스냅샷을 찍는다.
* 커밋 메세지는 지금 기록하는 이력을 충분히 잘 나타낼 수 있도록 작성
* `git log` 명령어를 통해 지금까지 기록된 커밋들을 확인 가능

## 3. log

> 커밋 히스토리

``` bash
$ git log
commit bbe02e60725d5a50d061eebdb13292e921a7e4e7 (HEAD -> master)
Author: dltmdgh1994 <dltmdgh1997@gmail.com>
Date:   Tue Dec 29 14:11:12 2020 +0900

    Third commit

commit d4b1e68539ceb5a8558c82bb868c61708d6b096f
Author: dltmdgh1994 <dltmdgh1997@gmail.com>
Date:   Tue Dec 29 11:43:28 2020 +0900

    Second commit!!!

commit 72a427a2dba31da3cf26766497e70dca5aff9cda
Author: dltmdgh1994 <dltmdgh1997@gmail.com>
Date:   Tue Dec 29 11:27:13 2020 +0900

    First commit!!!

$ git log --oneline
bbe02e6 (HEAD -> master) Third commit
d4b1e68 Second commit!!!
72a427a First commit!!!

$ git log -2
commit bbe02e60725d5a50d061eebdb13292e921a7e4e7 (HEAD -> master)
Author: dltmdgh1994 <dltmdgh1997@gmail.com>
Date:   Tue Dec 29 14:11:12 2020 +0900

    Third commit

commit d4b1e68539ceb5a8558c82bb868c61708d6b096f
Author: dltmdgh1994 <dltmdgh1997@gmail.com>
Date:   Tue Dec 29 11:43:28 2020 +0900

    Second commit!!!

$ git log --oneline -1
bbe02e6 (HEAD -> master) Third commit
```



## 4. git commit author 설정

git commit --global user.name "dltmdgh1994"

git commit --global user.email "dltmdgh1997@gmail.com"

``` bash
$ git config --global user.name
dltmdgh1994
```



## 5. 원격 저장소(remote repository) 활용 기초

> 다양한 원격저장소 서비스 중 Github를 기준

* 원격저장소 설정

  ``` bash
  $git remote add origin __url__
  ```

* push

  ```bash
  $git push -u origin master
  ```

* 설정된 원격저장소를 확인

  ```bash
  $ git remote -v
  origin  https://github.com/dltmdgh1994/multicampus.git (fetch)
  origin  https://github.com/dltmdgh1994/multicampus.git (push)
  ```

* 원격저장소 수정

  ```bash
  $ git remote set-url origin https://github.com/dltmdgh1994/multicampus.git
  ```

# git ignore

```bash
# gitignore를 생성
$ touch .gitignore
```

* 내부에 commit하고 싶지 않은 것들의 목록 작성