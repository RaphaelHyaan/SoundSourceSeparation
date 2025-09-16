import h5py

filename = "FastMNMF2_IP-param-M=4-S=3-F=513-K=64-init=twostep_50it-g=5e-10-bit=64-intv_norm=10-ID=00_02_hard.h5"

# 以只读方式打开 HDF5 文件
with h5py.File(filename, "r") as f:
    print("HDF5 文件中包含的数据集:")
    for key in f.keys():
        print(f"  {key} -> shape={f[key].shape}, dtype={f[key].dtype}")

    # 读取具体矩阵，例如 G_NM
    G_NM = f["G_NM"][:]
    print("G_NM 内容:\n", G_NM)

    # 读取 Q_FMM
    Q_FMM = f["Q_FMM"][:]
    # print("Q_FMM 内容:", Q_FMM)
    print("Q_FMM 形状:", Q_FMM.shape)