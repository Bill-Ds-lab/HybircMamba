import kagglehub
import os
import shutil


def download_dataset():
    datasets = {
        "German51K": "thanhsangtrn/german-51k", #GTSRB gốc
        "BelgiumTSC": "thanhsangtrn/belgiumtsc", #BelgiumTSC
        "GermanTrafficSign": "thanhsangtrn/german-trafic-sign" #GTSRB bị biến đổi của tác giả MambaTSR
    }

    destination_root = "data"
    os.makedirs(destination_root, exist_ok=True)

    for name, dataset_id in datasets.items():
        print("==================================Đang tải====================================="
        )

        destination = os.path.join(destination_root, name)
        if os.path.exists(destination) and os.listdir(destination):
            print(f"{name} đã tồn tại tại:")
            print(destination)
            print("Bỏ qua, không tải lại.")
            continue
        path = kagglehub.dataset_download(dataset_id)

        print(f"{name} đã tải về cache:")
        print(path)

        shutil.move(path, destination)

        print(f"Đã move {name} đến:")
        print(destination)

    print("=============================Tải xong================================="
    )



download_dataset()