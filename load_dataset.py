from datasets import load_dataset

# 指定项目文件夹路径
project_folder = '/mnt/afs/jingjinhao/project/VisuRiddles/datasets'

# 下载并加载数据集，同时将数据存储到指定文件夹
ds = load_dataset("yh0075/VisuRiddles", cache_dir=project_folder)