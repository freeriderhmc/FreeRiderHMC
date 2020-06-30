## Github repository clone method
> $ git clone https://github.com/freeriderhmc/Bin2PointCloud.git

## 작업공간 만들기
> 작업의 편의상 하나의 branch에서 하나의 push를 하는 것이 좋다.

> $ git branch <branch name>
  
> $ git checkout <branch name>
  
> * 현재 어떤 branch에 있는 지 확인 - $ git branch

## git push
> one-time setup

> $ git remote add origin {remote repository address}

> $ git add .

> $ git commit -m "commit content"

> $ git push origin <branch name>
  
> * 나의 local repository status 확인 - $ git status

## git rebase
> 내가 A버전의 snuzero repo master에서 branch를 생성해 A+B를 짜고 있는데, 누군가가 A+C를 push하면 내가 push할 수 없다. 이를 위해 master를 A+C로 업데이트하고 내 코드를 A+B+C로 만들어야 한다.

> $ git checkout master

> $ git pull

> $ git checkout <my_branch_name>

> $ git rebase master

> git repo version update를 자주 하지 않으면 rebase에서 심각한 문제가 생길 수 있으므로 자주 업데이트를 해주는 것이 좋다.
