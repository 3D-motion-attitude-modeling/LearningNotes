# Day 1（基础语法学习）

## Markdown基础语法

- 分段操作：`Enter`

- 换行但不分段：`Shift`+`Enter`

- 创建不同级别的标题：`#`

- 模块引用：`>`

  >Like this one

- 无序列表：`*` 

- 有序列表：`1.`

- 创建任务列表：`-[]`
  		- [ ] Like this one

- 引用代码块：```

- 键入公式：`$$`

- 创建表格：`|第一列|第二列|`

  | 第一列 | 第二列 |
  | ------ | ------ |
  |        |        |

- 分割线：`***`

- 生成目录：`[toc]`

- 嵌入Link：`[Explanation](Link)`

  [Like this](http://example.net/)

- 插入图片：`![Alt text][path "title"]`

  <img src="F:\Involution\大创—3D运动姿态\test.jpg" title="like this" style="zoom:50%;" />

- 斜体：`*text*`

- 加粗：`**text**`

- 嵌入式代码块：``

- 下标：`~`

​		H~2~

- 上标：`^`

   H^2^

- 高亮：`==text==`

   ==Like this==

   

## Git bash基础语法

### 盘符命令

- 切换盘符：`cd xxx/xxx`
- 查看当前文件夹路径：`pwd`
- 查看当前文件夹内容：`ls`

### 文件命令

- 将远端内容同步到本地仓库：`git clone <SSH_key>`

- 文件暂存至待提交区：`git add 文件名.文件类型`
  - 提交所有变化：`git add -A`
  - 提交被修改与被删除的文件，不包括新文件： `git add -u`
  - 提交新文件和被修改的文件，不包括删除（常用）： `git add .`
- 将暂存区文件提交：`git commit -m "注释"`
- 查看提交文件与仓库文件的差异：`git diff 文件名.文件类型`

- 将文件同步到远端仓库：`git push`
- 将远端文件更新到工作区文件：`git pull`
- 将远端文件取回到本地仓库，但不更新工作区文件： `git fetch`
  - 查看取回文件与工作区文件的差异：`git log -p FETCH_HEAD`
- 将本地仓库内容更新到工作区：`git merge`

### 分支命令

- 查看本地所有分支：`git branch`

- 查看远端所有分支：`git branch -r`

- 新建分支：`git branch <name>`

- 删除本地分支：`git branch -d <name>`

- 删除后更新到远端：`git push origin(远程主机名，一般为origin):<name>`

- 重命名本地分支：`git branch -m <oldname> <newname>`

- 切换到某个分支：`git checkout <name>`

  - 切换后，可将dev分支合并到master分支，并不影响dev分支的开发：

    `git merge dev`

### 远端，本地仓库，暂存区，工作区的关系图

![](F:\Involution\大创—3D运动姿态\git.jpg)