# Git 与 Slurm 工作规范

Git管理源码、配置、文档、来源清单和小型审阅证据；`/data/wushangge/CULS_LiteCut`管理所有新测试与程序运行数据。原有大目录通过索引统一访问，既有产物不移动、不删除、不改写。

```text
/data/wushangge/CULS_LiteCut/
  assets/
    inputs/       固定输入副本
    models/       预训练Student
    libraries/    冻结ASAP7库
    bin/          冻结ABC和gpuls
  history/
    imports/      Gate1/2/3及后续结果的实际归档副本
    links/        指向已有大型运行目录的链接
  runs/           后续每次测试或实验一个独立目录
  registries/     运行索引和管理记录
  cache/
  tmp/
```

新仓库采用独立的整合历史；原仓库的HEAD、脏工作区状态及所选tracked diff记录在`provenance/`，每个导入文件记录来源和哈希。不要将原仓库未提交修改遗漏为“仅复制HEAD”。初始整理完成后在`main`形成提交并打baseline tag。

后续一个实现或明确实验配置使用一个分支，例如`work/<topic>`，提交说明记录问题、改动和验证。一个结果必须能回到对应代码commit、输入/模型/library哈希和Slurm job；失败和中止结果也保留。不要修改已冻结基线或重用旧输出目录。Git远端尚未配置，本地提交不等于远程备份。

历史文件导入核验：

```bash
python -B scripts/verify_import.py
# 可选：同时检查原始来源有没有变化
python -B scripts/verify_import.py --check-originals
```

该工具校验导入基线的字节身份。未来有意修改源码时基线核验出现差异是预期现象，应通过Git记录；它不代替后续实现测试。

当一个实验已经确定并获得执行授权后，先提交对应源码，再准备目录：

```bash
python -B scripts/prepare_run.py --name <short-name> --purpose '<本次运行目的>'
```

准备工具要求干净Git工作区，生成`runs/<UTC时间>-<name>-<随机后缀>/`，包含`src/ input/ output/ logs/ tmp/ cache/ build/`和`run.json`。`src/`来自确定的Git commit，`input/`链接到数据区冻结资产；不会调用任何实验程序或提交Slurm。历史工具的旧路径需在本次运行配置/适配脚本中显式处理，不能直接执行原来的固定路径Slurm脚本。

每次真正提交前：

1. 按`slurm-data-workflow`检查当前partition、资源限制及环境。查看用户RUNNING和PENDING任务，合计不超过64 CPU、256 GiB RAM、2 GPU。
2. 将所需输入和运行源码放在本次`/data`目录。构建目录、TMPDIR、XDG_CACHE_HOME、模型输出和所有日志都指向本次目录。计算节点未必挂载`/home`。
3. 在该目录编写具体`job.slurm`，显式设置工作目录、stdout/stderr、time/CPU/memory及必要GPU；不沿用未核实的历史partition或资源配置。
4. `bash -n job.slurm`静态检查后从本次数据目录使用`sbatch --parsable job.slurm`提交。仅在已获交互allocation时使用`srun`。保存job ID、命令、环境和资源请求至`run.json`或伴随manifest。
5. 完成后记录`sacct`状态、退出码、资源用量、输出路径和哈希，完成CEC/timing等对应检查后再形成研究结论。Git只收录审阅过的小型结果和文档。

本轮整理不创建实际实验run，不选择新的研究方向，不提交Slurm任务。`scripts/import_history.py`是此次一次性导入过程记录，数据根目录已存在时会拒绝再次执行，避免覆盖历史。
