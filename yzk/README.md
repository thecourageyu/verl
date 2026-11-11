
gitlab folk & follow original project	

- step 1. To folk the project on gitlab or github website
- step 2. git clone & set remote upstream

```
$ git clone https://devops.foxconn.com/26778/agenticopsrl.git
$ git remote add upstream https://devops.foxconn.com/26776/agenticopsrl.git
$ git remote -v
```

```
origin  https://devops.foxconn.com/26778/agenticopsrl.git (fetch)
origin  https://devops.foxconn.com/26778/agenticopsrl.git (push)
upstream        https://devops.foxconn.com/26776/agenticopsrl.git (fetch)
upstream        https://devops.foxconn.com/26776/agenticopsrl.git (push)
```
- step 3. Get the updates from original project / author 

```
$ git fetch upstream
```

- step 4. merge or rebase

merge

```
$ git checkout main
$ git merge upstream/main
```

rebase
```
$ git checkout main
$ git rebase upstream/main
```

- step 5. push it to your folk
```
$ git push origin main
```
