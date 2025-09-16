import numpy as np

# ===================== 音频读取（audio loading）=====================
def load_wav_multichannel(path):
    """
    读取多通道 WAV，返回 float32 的 (T, C) 与采样率 sr。
    依赖 soundfile；若没有，可 pip install soundfile
    """
    import soundfile as sf
    data, sr = sf.read(path, always_2d=True)  # [T, C]
    data = data.astype(np.float32, copy=False)
    return data, sr

# ===================== 简易 STFT（short-time Fourier transform）=====================
def stft_multichannel(x_tc, n_fft=1024, hop=512, window='hann'):
    """
    输入: x_tc [T, C]，输出: X_fcn [F, C, N] (complex)
    F = n_fft//2+1, N = 帧数
    """
    T, C = x_tc.shape
    if window == 'hann':
        win = np.hanning(n_fft).astype(np.float32)
    else:
        raise ValueError("仅内置 hann 窗。")
    n_frames = 1 + (T - n_fft) // hop if T >= n_fft else 0
    if n_frames <= 0:
        raise RuntimeError("音频太短，无法分帧。请减小 n_fft 或使用更长音频。")
    F = n_fft // 2 + 1
    X_fcn = np.empty((F, C, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        s = i * hop
        frame = x_tc[s:s+n_fft, :] * win[:, None]      # [n_fft, C]
        spec = np.fft.rfft(frame, n=n_fft, axis=0)     # [F, C]
        X_fcn[:, :, i] = spec
    return X_fcn

# ===================== 逐频 SCM（per-frequency SCM）=====================
def scm_per_frequency(X_fcn, eps=0.0):
    """
    输入 X_fcn: [F, C, N]，输出 R_fcc: [F, C, C]（Hermitian）
    """
    F, C, N = X_fcn.shape
    R_fcc = np.empty((F, C, C), dtype=np.complex128)
    for f in range(F):
        X_cn = X_fcn[f, :, :]                  # [C, N]
        R = (X_cn @ X_cn.conj().T) / N         # [C, C]
        if eps > 0:
            R = R + eps * np.eye(C, dtype=R.dtype)
        R_fcc[f] = R
    return R_fcc

# ===================== PCA/特征分解（eigendecomposition）=====================
def pca_hermitian(R_cc):
    """
    对 Hermitian 矩阵做特征分解（PCA），返回 U, D（按特征值从大到小排序）
    R = U D U^H
    """
    # eigh 适用于 Hermitian，返回实特征值与复特征向量
    evals, evecs = np.linalg.eigh(R_cc)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    D = np.diag(evals)
    U = evecs
    return U, D

# ===================== 由主特征向量估计 DOA（theta） =====================
def estimate_theta_from_principal_u(u1, f_hz, d, c=343.0):
    """
    使用 ULA 模型 a_f(θ)[m] = exp(-j*2π f/c * m d cosθ),
    由主特征向量 u1 估计 θ（度）。
    做法：
      1) 以第0通道作参考，去除公共相位：u_norm[m] = u1[m]/u1[0]
      2) 相位展开：phi_m = unwrap(angle(u_norm[m]))
      3) 线性拟合 phi_m ~ slope * m + b，其中 slope ≈ - 2π f d / c * cosθ
      4) cosθ = - slope * c / (2π f d)
    """
    M = u1.shape[0]
    # 归一化去公共相位影响
    u_norm = u1 / (u1[0] if u1[0] != 0 else 1.0)
    m = np.arange(M, dtype=np.float64)
    phi = np.unwrap(np.angle(u_norm))  # [M]
    # 线性拟合（最小二乘）
    slope, intercept = np.polyfit(m, phi, 1)
    cos_theta = - slope * c / (2.0 * np.pi * f_hz * d)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(cos_theta)))
    return theta_deg, cos_theta, slope, intercept, phi

# ===================== 工具：打印矩阵 =====================
def print_matrix(M, name, max_rows=6, precision=4, scientific=True):
    np.set_printoptions(precision=precision, suppress=not scientific)
    print(f"\n{name} 形状: {M.shape}")
    to_show = M if M.shape[0] <= max_rows else M[:max_rows]
    print(to_show)

# ===================== 示例主流程 =====================
def main():
    # --------- 需要你修改的参数 ----------
    wav_path = "src/separation/02_single.wav"      # 你的4通道 WAV 路径
    target_f_hz = 1000.0              # 选择估计的频率（Hz）
    d = 0.05                          # 阵元间距（m），例如 5 cm 的 ULA
    c = 343.0                         # 声速（m/s）
    n_fft, hop = 1024, 512
    # -----------------------------------

    # 1) 读取音频
    x_tc, sr = load_wav_multichannel(wav_path)
    T, C = x_tc.shape
    if C != 4:
        raise RuntimeError(f"期望4通道，实际 {C} 通道。请提供4通道 WAV。")
    print(f"采样率: {sr} Hz, 时长: {T/sr:.2f} s, 通道数: {C}")

    # 2) STFT
    X_fcn = stft_multichannel(x_tc, n_fft=n_fft, hop=hop, window='hann')  # [F,C,N]
    F = X_fcn.shape[0]
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)  # [F]
    # 选择最接近 target_f_hz 的频点
    f_idx = int(np.argmin(np.abs(freqs - target_f_hz)))
    f_sel = float(freqs[f_idx])
    print(f"选用频率: {f_sel:.2f} Hz (最接近 {target_f_hz} Hz)")

    # 3) 逐频SCM并取该频点
    R_fcc = scm_per_frequency(X_fcn, eps=0.0)  # [F,C,C]
    R = R_fcc[f_idx]                            # [C,C]
    print_matrix(R, f"SCM @ {f_sel:.2f} Hz", precision=6, scientific=True)

    # 4) PCA/特征分解：R = U D U^H
    U, D = pca_hermitian(R)
    # 验证对角化
    D_check = U.conj().T @ R @ U
    print_matrix(D, "特征值对角矩阵 D", precision=6, scientific=True)
    print_matrix(U, "特征向量矩阵 U（列为特征向量）", precision=6, scientific=True)
    print_matrix(D_check, "U^H R U（应近似对角）", precision=6, scientific=True)

    # 5) 取第一主特征向量并估计 θ
    u1 = U[:, 0]  # 主分量（对应最大特征值）
    theta_deg, cos_theta, slope, intercept, phi = estimate_theta_from_principal_u(
        u1, f_sel, d, c=c
    )

    # 6) 打印结果与混叠检查
    alias_limit = c / (2.0 * d)  # 频率上限，超过则可能空间混叠
    print(f"\n估计结果（DOA, direction of arrival）@ {f_sel:.2f} Hz：")
    print(f"- 估计角度 θ（degrees）      : {theta_deg:.2f}")
    print(f"- 估计 cosθ                  : {cos_theta:.6f}")
    print(f"- 相位-阵元索引斜率 slope     : {slope:.6f} rad/index")
    print(f"- 线性拟合截距 intercept      : {intercept:.6f} rad（应接近0）")
    print(f"- 空间混叠频率上限 c/(2d)     : {alias_limit:.1f} Hz")
    if f_sel > alias_limit:
        print("⚠ 警告：当前频率超过 c/(2d)，可能存在空间混叠（角度歧义）。")

    # 7) 可选：打印相位序列（相对于第0阵元）
    print("\n相位展开 φ_m（相对第0通道，rad）：", np.array2string(phi, precision=4))

if __name__ == "__main__":
    main()
