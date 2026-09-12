import kagglehub
import os
import shutil

def download_dataset():
    path = kagglehub.dataset_download(
        "thanhsangtrn/german-51k"
    )

    print("Dataset đã tải về:")
    print(path)

    destination = (
        "data/German_51k"
    )

    os.makedirs(destination, exist_ok=True)

    shutil.copytree(
        path,
        destination,
        dirs_exist_ok=True
    )

    print("Đã copy dataset vào:")
    print(destination)
    return destination
download_dataset()