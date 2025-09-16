#! /usr/bin/env python3
# coding: utf-8

import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from Base import EPS, MIC_INDEX, Base, MultiSTFT


def deconv1d_freq(Wm_NFprimeK, k_L, stride=5, pad=0, xp=np):
    """
    频率轴转置卷积 (ConvTranspose1D)
    Wm_NFprimeK: (N, F', K)
    k_L: (L,) 卷积核（非负）
    返回: (N, F, K)，其中 F = (F'-1)*stride - 2*pad + L
    """
    N, Fp, K = Wm_NFprimeK.shape
    L = len(k_L)
    F = (Fp - 1) * stride - 2 * pad + L
    W_tilde = xp.zeros((N, F, K), dtype=Wm_NFprimeK.dtype)

    for fprime in range(Fp):
        start = fprime * stride - pad
        for p in range(L):
            f = start + p
            if 0 <= f < F:
                W_tilde[:, f, :] += Wm_NFprimeK[:, fprime, :] * k_L[p]
    return W_tilde + EPS

def deconv1d_T(W_ratio_NFK, k_L, Fp, stride=5, pad=0, xp=np):
    """
    转置卷积的伴随算子（相当于普通卷积），把 (N,F,K) 回传到 (N,F',K)
    W_ratio_NFK: (N, F, K)
    k_L: (L,) 卷积核
    Fp: 原始 F'
    返回: (N, F', K)
    """
    N, F, K = W_ratio_NFK.shape
    L = len(k_L)
    Wm_ratio = xp.zeros((N, Fp, K), dtype=W_ratio_NFK.dtype)

    for fprime in range(Fp):
        start = fprime * stride - pad
        for p in range(L):
            f = start + p
            if 0 <= f < F:
                Wm_ratio[:, fprime, :] += W_ratio_NFK[:, f, :] * k_L[p]
    return Wm_ratio + EPS

class FastMNMF2_WM(Base):
    """
    The blind souce separation using FastMNMF2

    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts SCMs to diagonal matrices
    G_NM: diagonal elements of the diagonalized SCMs
    W_NFK: basis vectors
    H_NKT: activations
    PSD_NFT: power spectral densities
    Qx_power_FTM: power spectra of Q_FMM times X_FTM
    Y_FTM: sum of (PSD_NFT x G_NM) over all sources
    """

    def __init__(
        self,
        n_source,
        n_basis=8,
        init_SCM="twostep",
        algo="IP",
        n_iter_init=30,
        g_eps=5e-2,
        interval_norm=10,
        n_bit=64,
        xp=np,
        seed=0,
        init_angle_deg = 10,
    ):
        """Initialize FastMNMF2

        Parameters:
        -----------
            n_source: int
                The number of sources.
            n_basis: int
                The number of bases for the NMF-based source model.
            init_SCM: str ('circular', 'obs', 'twostep')
                How to initialize SCM.
                'obs' is for the case that one speech is dominant in the mixture.
            algo: str (IP, ISS)
                How to update Q.
            n_iter_init: int
                The number of iteration for the first step in 'twostep' initialization.
            xp : numpy or cupy
        """
        super().__init__(xp=xp, n_bit=n_bit, seed=seed)
        self.n_source = n_source
        self.n_basis = n_basis
        self.init_SCM = init_SCM
        self.g_eps = g_eps
        self.algo = algo
        self.interval_norm = interval_norm
        self.n_iter_init = n_iter_init
        self.save_param_list += ["Wm_NFprimeK","k_L", "H_NKT", "G_NM", "Q_FMM"]
        self.init_angle_deg = init_angle_deg

        self.Fp = 101   # 低分辨率频点数
        self.stride = 5
        self.L = 13
        self.pad = 0

        self.Wm_NFprimeK = None
        self.k_L = None

        # self.init_source_model()

        if self.algo == "IP":
            self.method_name = "FastMNMF2_IP"
        elif "ISS" in algo:
            self.method_name = "FastMNMF2_ISS"
        else:
            raise ValueError("algo must be IP or ISS")

    def __str__(self):
        init = f"twostep_{self.n_iter_init}it" if self.init_SCM == "twostep" else self.init_SCM
        filename_suffix = (
            f"M={self.n_mic}-S={self.n_source}-F={self.n_freq}-K={self.n_basis}"
            f"-init={init}-g={self.g_eps}-bit={self.n_bit}-intv_norm={self.interval_norm}"
        )
        if hasattr(self, "file_id"):
            filename_suffix += f"-ID={self.file_id}"
        return filename_suffix
    
    def init_source_model(self):
        # 初始化小 Wm 与 H、k
        self.Wm_NFprimeK = self.xp.random.rand(self.n_source, self.Fp, self.n_basis).astype(self.TYPE_FLOAT)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)
        # 卷积核初始化为非负并 L1 归一
        k = self.xp.ones(self.L, dtype=self.TYPE_FLOAT)
        k /= k.sum() + EPS
        self.k_L = k

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super().load_spectrogram(X_FTM, sample_rate=sample_rate)
        if self.algo == "IP":
            self.XX_FTMM = self.xp.einsum("fti, ftj -> ftij", self.X_FTM, self.X_FTM.conj())

    # def init_source_model(self):
    #     self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
    #     self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)

    def init_spatial_model(self):
        self.start_idx = 0
        self.Q_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
        self.G_NM = self.xp.ones([self.n_source, self.n_mic], dtype=self.TYPE_FLOAT) * self.g_eps
        # for m in range(self.n_source):
        #     self.G_NM[m % self.n_source, m] = 1
        # self.G_NM[0,0] = 0.6
        # self.G_NM[0,1] = 0.4
        self.G_NM[0,0] = 0.8
        self.G_NM[0,1] = 0.2





        if "circular" in self.init_SCM:
            pass
        elif "obs" in self.init_SCM:
            if hasattr(self, "XX_FTMM"):
                XX_FMM = self.XX_FTMM.sum(axis=1)
            else:
                XX_FMM = self.xp.einsum("fti, ftj -> fij", self.X_FTM, self.X_FTM.conj())
            _, eig_vec_FMM = self.xp.linalg.eigh(XX_FMM)
            eig_vec_FMM = eig_vec_FMM[:, :, ::-1]
            self.Q_FMM = self.xp.asarray(eig_vec_FMM).transpose(0, 2, 1).conj()

        elif "angle" == self.init_SCM:

            # if self.n_iter_init >= self.n_iter:
            #     print(
            #         "\n------------------------------------------------------------------\n"
            #         f"Warning: n_iter_init must be smaller than n_iter (= {self.n_iter}).\n"
            #         f"n_iter_init is changed from {self.n_iter_init} to {self.n_iter // 3}"
            #         "\n------------------------------------------------------------------\n"
            #     )
            #     self.n_iter_init = self.n_iter // 3

            # self.start_idx = self.n_iter_init

            # separater_init = FastMNMF2(
            #     n_source=self.n_source,
            #     n_basis=2,
            #     init_SCM="circular",
            #     xp=self.xp,
            #     n_bit=self.n_bit,
            #     g_eps=self.g_eps,
            # )
            # separater_init.load_spectrogram(self.X_FTM, self.sample_rate)
            # separater_init.solve(n_iter=self.start_idx, save_wav=False)

            # self.Q_FMM = separater_init.Q_FMM
            # self.G_NM = separater_init.G_NM

            # 读取/默认阵列与物理参数
            c = float(getattr(self, "sound_speed", 343.0))               # 声速(m/s)
            d = float(getattr(self, "array_spacing", 0.05))             # 阵元间距(m)
            if not hasattr(self, "init_angle_deg"):
                raise ValueError("init_SCM='angle' 需要设置 self.init_angle_deg（单位：度）")
            theta = float(self.init_angle_deg) * self.xp.pi / 180.0      # 转弧度

            F, M = self.n_freq, self.n_mic
            # 由 n_freq 反推 n_fft（rfft 情况）：n_freq = n_fft//2 + 1
            n_fft = int((F - 1) * 2)
            # 频率轴(Hz)，与 rfftfreq 一致
            # 注：self.xp 可能是 cupy；后续 QR 用 numpy 更稳，先在 CPU 上构造频率
            import numpy as _np
            freqs = _np.fft.rfftfreq(n_fft, d=1.0 / float(self.sample_rate))  # 长度应为 F
            if len(freqs) != F:
                raise ValueError(f"频点数不匹配: n_freq={F}, rfftfreq 得到 {len(freqs)}")


            # 针对每个频点，构造导向向量 a_f 并放入 P_f 的第0列；其余用 QR 补全为酉矩阵
            target_col = 3 # 默认只替换第0列
            for f_idx in range(self.n_freq):
                f_hz = freqs[f_idx]
                m_idx = _np.arange(self.n_mic, dtype=float)
                phase = -2.0 * _np.pi * f_hz * d * _np.cos(theta) * m_idx / c
                a_f = _np.exp(1j * phase)
                a_f /= _np.linalg.norm(a_f) + 1e-12
                self.Q_FMM[f_idx, :, target_col] = self.xp.asarray(a_f, dtype=self.TYPE_COMPLEX)
        
        elif "twoangle" == self.init_SCM:
            
            if self.n_iter_init >= self.n_iter:
                print(
                    "\n------------------------------------------------------------------\n"
                    f"Warning: n_iter_init must be smaller than n_iter (= {self.n_iter}).\n"
                    f"n_iter_init is changed from {self.n_iter_init} to {self.n_iter // 3}"
                    "\n------------------------------------------------------------------\n"
                )
                self.n_iter_init = self.n_iter // 3

            self.start_idx = self.n_iter_init

            separater_init = FastMNMF2_WM(
                n_source=self.n_source,
                n_basis=2,
                init_SCM="angle",
                xp=self.xp,
                n_bit=self.n_bit,
                g_eps=self.g_eps,
                init_angle_deg=self.init_angle_deg
            )
            separater_init.load_spectrogram(self.X_FTM, self.sample_rate)
            separater_init.solve(n_iter=self.start_idx, save_wav=False)

            self.Q_FMM = separater_init.Q_FMM
            self.G_NM = separater_init.G_NM

        elif "twostep" == self.init_SCM:
            if self.n_iter_init >= self.n_iter:
                print(
                    "\n------------------------------------------------------------------\n"
                    f"Warning: n_iter_init must be smaller than n_iter (= {self.n_iter}).\n"
                    f"n_iter_init is changed from {self.n_iter_init} to {self.n_iter // 3}"
                    "\n------------------------------------------------------------------\n"
                )
                self.n_iter_init = self.n_iter // 3

            self.start_idx = self.n_iter_init

            separater_init = FastMNMF2_WM(
                n_source=self.n_source,
                n_basis=2,
                init_SCM="circular",
                xp=self.xp,
                n_bit=self.n_bit,
                g_eps=self.g_eps,
            )
            separater_init.load_spectrogram(self.X_FTM, self.sample_rate)
            separater_init.solve(n_iter=self.start_idx, save_wav=False)

            self.Q_FMM = separater_init.Q_FMM
            self.G_NM = separater_init.G_NM
        else:
            raise ValueError("init_SCM should be circular, obs, or twostep.")

        self.G_NM /= self.G_NM.sum(axis=1)[:, None]
        self.normalize()

    def calculate_Qx(self):
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        self.Qx_power_FTM = self.xp.abs(self.Qx_FTM) ** 2

    def calculate_PSD(self):
        self.W_tilde_NFK = deconv1d_freq(self.Wm_NFprimeK, self.k_L,
                                     stride=self.stride, pad=self.pad, xp=self.xp)
        self.PSD_NFT = self.W_tilde_NFK @ self.H_NKT + EPS

    def calculate_Y(self):
        self.Y_FTM = self.xp.einsum("nft, nm -> ftm", self.PSD_NFT, self.G_NM) + EPS

    def update(self):
        self.update_WH()
        self.update_kernel()
        self.update_G()
        if self.algo == "IP":
            self.update_Q_IP()
        else:
            self.update_Q_ISS()
        if self.it % self.interval_norm == 0:
            self.normalize()
        else:
            self.calculate_Qx()
        return 

    def update_WH(self):
        tmp1_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)

        Num_NFK = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp1_NFT)
        Den_NFK = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp2_NFT)

        Num_NFprimeK = deconv1d_T(Num_NFK, self.k_L, self.Fp,
                              stride=self.stride, pad=self.pad, xp=self.xp)
        Den_NFprimeK = deconv1d_T(Den_NFK, self.k_L, self.Fp,
                              stride=self.stride, pad=self.pad, xp=self.xp)
        self.Wm_NFprimeK *= self.xp.sqrt(Num_NFprimeK / (Den_NFprimeK + EPS))

        self.calculate_PSD()
        self.calculate_Y()

        tmp1_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        numerator   = self.xp.einsum("nfk, nft -> nkt", self.W_tilde_NFK, tmp1_NFT)
        denominator = self.xp.einsum("nfk, nft -> nkt", self.W_tilde_NFK, tmp2_NFT)
        self.H_NKT *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

    def update_kernel(self):
        # 1) 构造与更新 Wm 相同的 Num/Den（基于当前 H 与 (tmp1,tmp2)）
        tmp1_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        Num_NFK = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp1_NFT)
        Den_NFK = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp2_NFT)

        # 2) 统计每个 tap 的投影：num_k[p] = <S_p(Wm), Num>, den_k[p] = <S_p(Wm), Den>
        N, Fp, K = self.Wm_NFprimeK.shape
        _, F, _ = Num_NFK.shape
        L = self.L
        num_k = self.xp.zeros(L, dtype=self.TYPE_FLOAT)
        den_k = self.xp.zeros(L, dtype=self.TYPE_FLOAT)

        # 累加：S_p(Wm) 在频轴位置 f = f'*stride - pad + p
        fprime_idx = self.xp.arange(Fp)
        for p in range(L):
            f_idx = fprime_idx * self.stride - self.pad + p  # 对应输出频点
            mask = (f_idx >= 0) & (f_idx < F)
            if mask.any():
                vfprime = fprime_idx[mask]
                vf = f_idx[mask]
                # 取子张量并相乘累加
                Wm_sub = self.Wm_NFprimeK[:, vfprime, :]  # (N, Fv, K)
                Num_sub = Num_NFK[:, vf, :]               # (N, Fv, K)
                Den_sub = Den_NFK[:, vf, :]               # (N, Fv, K)
                num_k[p] = (Wm_sub * Num_sub).sum()
                den_k[p] = (Wm_sub * Den_sub).sum()

        # 3) 乘法更新 + L1 归一化，并把尺度回吸收进 Wm
        self.k_L *= self.xp.sqrt((num_k + EPS) / (den_k + EPS))
        s = self.k_L.sum() + EPS
        self.k_L /= s
        self.Wm_NFprimeK *= s  # 归一化核后，用等效尺度回吸收避免能量漂移

        # 4) 刷新
        self.calculate_PSD()
        self.calculate_Y()

    def update_G(self):
        numerator = self.xp.einsum("nft, ftm -> nm", self.PSD_NFT, self.Qx_power_FTM / (self.Y_FTM**2))
        denominator = self.xp.einsum("nft, ftm -> nm", self.PSD_NFT, 1 / self.Y_FTM)
        self.G_NM *= self.xp.sqrt(numerator / denominator)
        self.calculate_Y()

    def update_Q_IP(self):
        for m in range(self.n_mic):
            V_FMM = self.xp.einsum("ftij, ft -> fij", self.XX_FTMM, 1 / self.Y_FTM[..., m]) / self.n_time
            tmp_FM = self.xp.linalg.inv(self.Q_FMM @ V_FMM)[..., m]
            self.Q_FMM[:, m] = (
                tmp_FM / self.xp.sqrt(self.xp.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[:, None]
            ).conj()

    def update_Q_ISS(self):
        for m in range(self.n_mic):
            QxQx_FTM = self.Qx_FTM * self.Qx_FTM[:, :, m, None].conj()
            V_tmp_FxM = (QxQx_FTM[:, :, m, None] / self.Y_FTM).mean(axis=1)
            V_FxM = (QxQx_FTM / self.Y_FTM).mean(axis=1) / V_tmp_FxM
            V_FxM[:, m] = 1 - 1 / self.xp.sqrt(V_tmp_FxM[:, m])
            self.Qx_FTM -= self.xp.einsum("fm, ft -> ftm", V_FxM, self.Qx_FTM[:, :, m])
            self.Q_FMM -= self.xp.einsum("fi, fj -> fij", V_FxM, self.Q_FMM[:, m])

    def normalize(self):
        # 1) 归一化 Q
        phi_F = (self.xp.einsum("fij, fij -> f", self.Q_FMM, self.Q_FMM.conj()).real / self.n_mic)
        self.Q_FMM /= self.xp.sqrt(self.xp.maximum(phi_F, EPS))[:, None, None]

        # 2) 核 k_L 做 L1 归一化，并把尺度回吸收到 Wm
        s_k = self.k_L.sum() + EPS
        self.k_L /= s_k
        # 标量就地乘法安全
        self.Wm_NFprimeK *= s_k

        # 3) G 的行归一化（每个源各通道权重和为 1），尺度回吸收到 Wm
        mu_N = self.G_NM.sum(axis=1, keepdims=False) + EPS          # (N,)
        mu_N = mu_N.astype(self.TYPE_FLOAT, copy=False).reshape(-1, 1, 1)  # (N,1,1)
        self.G_NM = self.G_NM / mu_N.reshape(-1, 1)                  # 广播到 (N,M)
        # ❗避免就地广播，改为新数组赋值
        self.Wm_NFprimeK = self.Wm_NFprimeK * mu_N                   # (N,F',K) = (N,F',K) * (N,1,1)

        # 4) 计算 W̃ 并做 (n,k) 配对归一化：sum_f W̃_{n f k} = 1
        self.W_tilde_NFK = deconv1d_freq(
            self.Wm_NFprimeK, self.k_L,
            stride=self.stride, pad=self.pad,
            xp=self.xp
        )  # (N, F, K)

        nu_NK = self.xp.maximum(self.W_tilde_NFK.sum(axis=1), EPS)   # (N,K)
        nu_NK = nu_NK.astype(self.TYPE_FLOAT, copy=False)

        # ❗避免就地广播：用新数组赋值
        self.Wm_NFprimeK = self.Wm_NFprimeK / nu_NK[:, None, :]      # (N,F',K) / (N,1,K)
        self.H_NKT       = self.H_NKT *  nu_NK[:, :, None]           # (N,K,T) * (N,K,1)

        # 5) 刷新统计量
        self.calculate_Qx()
        self.calculate_PSD()
        self.calculate_Y()
    def separate(self, mic_index=MIC_INDEX):
        Y_NFTM = self.xp.einsum("nft, nm -> nftm", self.PSD_NFT, self.G_NM)
        self.Y_FTM = Y_NFTM.sum(axis=0)
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        Qinv_FMM = self.xp.linalg.inv(self.Q_FMM)

        self.separated_spec = self.xp.einsum(
            "fj, ftj, nftj -> nft", Qinv_FMM[:, mic_index], self.Qx_FTM / self.Y_FTM, Y_NFTM
        )
        return self.separated_spec

    def calculate_log_likelihood(self):
        log_likelihood = (
            -(self.Qx_power_FTM / self.Y_FTM + self.xp.log(self.Y_FTM)).sum()
            + self.n_time * (self.xp.log(self.xp.linalg.det(self.Q_FMM @ self.Q_FMM.transpose(0, 2, 1).conj()))).sum()
        ).real
        return log_likelihood

    def load_param(self, filename):
        super().load_param(filename)

        self.n_source, self.n_freq, self.n_basis = self.W_NFK.shape
        _, _, self.n_time = self.H_NKT


if __name__ == "__main__":
    import soundfile as sf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fname",default='src/separation/canonrock.wav', type=str, help="filename of the multichannel observed signals")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_source", type=int, default=2, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=64, help="number of basis")
    parser.add_argument("--n_iter_init", type=int, default=50, help="nujmber of iteration used in twostep init")
    parser.add_argument(
        "--init_SCM",
        type=str,
        default="twostep",
        help="circular, obs (only for enhancement), twostep",
    )
    parser.add_argument("--n_iter", type=int, default=150, help="number of iteration")
    parser.add_argument("--g_eps", type=float, default=5e-10, help="minumum value used for initializing G_NM")
    parser.add_argument("--n_mic", type=int, default=2, help="number of microphone")
    parser.add_argument("--n_bit", type=int, default=64, help="number of microphone")
    parser.add_argument("--algo", type=str, default="IP", help="the method for updating Q")
    parser.add_argument("--init_angle_deg", type=int,default=40)
    # parser.add_argument("--init_angle_deg", type=int,default=120.91)
    parser.add_argument("--save_path",type = str,default='./')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        try:
            import cupy as xp

            print("Use GPU " + str(args.gpu))
            xp.cuda.Device(args.gpu).use()
        except ImportError:
            print("Warning: cupy is not installed. 'gpu' argument should be set to -1. Switched to CPU.\n")
            import numpy as xp

    separater = FastMNMF2_WM(
        n_source=args.n_source,
        n_basis=args.n_basis,
        xp=xp,
        init_SCM=args.init_SCM,
        n_bit=args.n_bit,
        algo=args.algo,
        n_iter_init=args.n_iter_init,
        g_eps=args.g_eps,
        init_angle_deg = args.init_angle_deg
    )

    wav, sample_rate = sf.read(args.input_fname)
    wav /= np.abs(wav).max() * 1.2
    M = min(len(wav), args.n_mic)
    spec_FTM = MultiSTFT(wav[:, :M], n_fft=args.n_fft)

    separater.file_id = args.input_fname.split("/")[-1].split(".")[0]
    separater.load_spectrogram(spec_FTM, sample_rate)

    separater.solve(
        n_iter=args.n_iter,
        save_dir=args.save_path,
        save_likelihood=True,
        save_wav=True,
        save_param=True,
        interval_save=5,
        save_waveplot=True,
    )
