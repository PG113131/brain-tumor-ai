import shutil
import kagglehub

# Download latest version of the Brain Tumor MRI Dataset
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

# Move/copy directly to your local data folder
shutil.copytree(path, "data/dataset", dirs_exist_ok=True)
print("Dataset downloaded and placed in data/dataset!")