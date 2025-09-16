#! /usr/bin/env python3
# coding: utf-8

import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from Base import EPS, MIC_INDEX, Base, MultiSTFT


class FastMNMF2(Base):
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
        target_col = 0,
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
        self.save_param_list += ["W_NFK", "H_NKT", "G_NM", "Q_FMM"]
        self.init_angle_deg = init_angle_deg
        self.target_col = target_col

        self.n_fft = 1024
        self.fs_hz = 48000

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
    
    def _steering_vector_deg(self, f_hz, theta_deg, d_list, c=343.0, A=1.0):
        """
        生成单频导向向量(steering vector):
            a_f[m] = A * exp(-j * (2π f / c) * d[m] * cos(theta))
        输入/输出均使用 self.xp（numpy/cupy 兼容）。
        """
        xp = self.xp
        d = xp.asarray(d_list, dtype=xp.float64).reshape(-1)
        theta_rad = xp.deg2rad(xp.asarray(theta_deg, dtype=xp.float64))
        k = 2.0 * xp.pi * f_hz / c
        phase = -k * d * xp.cos(theta_rad)
        a_f = A * xp.exp(1j * phase)
        return a_f.astype(self.TYPE_COMPLEX)

    def init_qfmm_with_angle(self,
                            m_col,
                            theta_deg,
                            d_list=None,
                            c=343.0,
                            A=1.0,
                            q_fmm_in=None):
        """
        **按角度( degree )初始化 Q_FMM（Frequency-by-frequency ）**
        使得 P_f = (Q_f)^(-1) 的第 m_col 列等于导向向量 a_f。

        参数:
            m_col     : int, 指定嵌入导向向量的列索引（0..M-1）
            theta_deg : float 或 array-like(长度==n_freq), 角度(度, degree)
            d_list    : list/ndarray, 阵元坐标(m)。默认: 等间距 ULA => xp.arange(M)*self.d_m
            c         : float, 声速(m/s)
            A         : float, 幅度(Amplitude)
            q_fmm_in  : ndarray (F,M,M), 若提供，在其基础上覆盖；否则从单位阵开始构造

        返回:
            Q_FMM_new : (F,M,M) complex, 初始化后的 Q_FMM
        """
        xp = self.xp
        F, M = self.n_freq, self.n_mic
        assert 0 <= m_col < M, f"m_col 超界: {m_col} not in [0,{M-1}]"

        # 频率轴（与 rFFT 对齐）
        # 若已有 self.freqs_hz 列表则直接使用，否则依据 fs/n_fft 计算
        if hasattr(self, "freqs_hz") and self.freqs_hz is not None:
            freqs_hz = xp.asarray(self.freqs_hz, dtype=xp.float64)
            assert freqs_hz.shape[0] == F, "self.freqs_hz 长度与 self.n_freq 不一致"
        else:
            freqs_hz = xp.fft.rfftfreq(self.n_fft, d=1.0 / self.fs_hz).astype(xp.float64)
            assert freqs_hz.shape[0] == F, "由 fs/n_fft 计算的频点数与 self.n_freq 不一致"

        # 阵元坐标
        if d_list is None:
            d_list = xp.arange(M, dtype=xp.float64) * float(self.d_m)
        else:
            d_list = xp.asarray(d_list, dtype=xp.float64).reshape(-1)
            assert d_list.shape[0] == M, "d_list 长度必须等于 self.n_mic"

        # 角度序列
        theta_seq = xp.asarray(theta_deg, dtype=xp.float64)
        if theta_seq.ndim == 0:
            theta_seq = xp.full((F,), float(theta_seq), dtype=xp.float64)
        else:
            assert theta_seq.shape[0] == F, "theta_deg 为序列时，其长度必须等于 n_freq"

        # 起始 Q_FMM
        if q_fmm_in is None:
            Q_FMM = xp.tile(xp.eye(M, dtype=self.TYPE_COMPLEX), (F, 1, 1))
        else:
            Q_FMM = xp.asarray(q_fmm_in, dtype=self.TYPE_COMPLEX)
            assert Q_FMM.shape == (F, M, M), "q_fmm_in 形状必须为 (F,M,M)"

        # 单位向量 e_m 与恒等矩阵 I
        e_m = xp.zeros((M,), dtype=self.TYPE_COMPLEX)
        e_m[m_col] = 1.0 + 0j
        I = xp.eye(M, dtype=self.TYPE_COMPLEX)

        # 逐频点构造 Q_f = P_f^{-1}，其中 P_f 的第 m_col 列为 a_f
        # 使用 Sherman–Morrison: Q_f = I - ( (a - e_m) e_m^T ) / a[m]
        eps = 1e-12
        for fi in range(F):
            f_hz = float(freqs_hz[fi])
            th_deg = float(theta_seq[fi])

            a_f = self._steering_vector_deg(f_hz, th_deg, d_list, c=c, A=A)  # (M,)
            a_m = a_f[m_col]

            if xp.abs(a_m) < eps:
                # 极罕见：若 a[m]==0，退化到显式逆（稳妥）
                P_f = I.copy()
                P_f[:, m_col] = a_f
                Q_f = xp.linalg.inv(P_f)
            else:
                u = a_f - e_m                      # (M,)
                # 组装: Q_f = I - (u e_m^T) / a_m
                # 外积 u e_m^T：形状 (M,M)，仅第 m_col 列非零
                Q_f = I - (u[:, None] @ e_m[None, :]) / a_m

            Q_FMM[fi, :, :] = Q_f

        return Q_FMM

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super().load_spectrogram(X_FTM, sample_rate=sample_rate)
        if self.algo == "IP":
            self.XX_FTMM = self.xp.einsum("fti, ftj -> ftij", self.X_FTM, self.X_FTM.conj())

    def init_source_model(self):
        self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)

    def init_spatial_model(self):
        self.start_idx = 0
        self.Q_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
        self.G_NM = self.xp.ones([self.n_source, self.n_mic], dtype=self.TYPE_FLOAT) * self.g_eps
        # for m in range(self.n_source):
        #     self.G_NM[m % self.n_source, m] = 1
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


            # 针对每个频点，构造导向向量 a_f 并放入 P_f 的第0列；其余用 QR 补全为酉矩阵
            target_col = self.target_col # 默认只替换第0列
            self.Q_FMM = self.init_qfmm_with_angle(
                m_col=target_col,
                theta_deg=self.init_angle_deg,
                d_list=[0,0.05,0.15,0.2],
                q_fmm_in=self.Q_FMM
            )

        elif "twoangle" == self.init_SCM:
            
            if self.n_iter_init >= self.n_iter:
                self.n_iter_init = self.n_iter // 3

            self.start_idx = self.n_iter_init

            separater_init = FastMNMF2(
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
                    "/n------------------------------------------------------------------/n"
                    f"Warning: n_iter_init must be smaller than n_iter (= {self.n_iter})./n"
                    f"n_iter_init is changed from {self.n_iter_init} to {self.n_iter // 3}"
                    "/n------------------------------------------------------------------/n"
                )
                self.n_iter_init = self.n_iter // 3

            self.start_idx = self.n_iter_init

            separater_init = FastMNMF2(
                n_source=self.n_source,
                n_basis=2,
                init_SCM="circular",
                xp=self.xp,
                n_bit=self.n_bit,
                g_eps=self.g_eps,
                # seed=1
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
        self.PSD_NFT = self.W_NFK @ self.H_NKT + EPS

    def calculate_Y(self):
        self.Y_FTM = self.xp.einsum("nft, nm -> ftm", self.PSD_NFT, self.G_NM) + EPS

    def update(self):
        self.update_WH()
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
        numerator = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp1_NFT)
        denominator = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp2_NFT)
        self.W_NFK *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

        tmp1_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        numerator = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, tmp1_NFT)
        denominator = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, tmp2_NFT)
        self.H_NKT *= self.xp.sqrt(numerator / denominator)
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
        phi_F = self.xp.einsum("fij, fij -> f", self.Q_FMM, self.Q_FMM.conj()).real / self.n_mic
        self.Q_FMM /= self.xp.sqrt(phi_F)[:, None, None]
        self.W_NFK /= phi_F[None, :, None]

        mu_N = self.G_NM.sum(axis=1)
        self.G_NM /= mu_N[:, None]
        self.W_NFK *= mu_N[:, None, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        self.H_NKT *= nu_NK[:, :, None]

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
    parser.add_argument("--input_fname",default='src/separation/00_02_normal_3.wav', type=str, help="filename of the multichannel observed signals")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_source", type=int, default=3, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=64, help="number of basis")
    parser.add_argument("--n_iter_init", type=int, default=50, help="number of iteration used in twostep init")
    parser.add_argument(
        "--init_SCM",
        type=str,
        default="angle",
        help="circular, obs (only for enhancement), twostep",
    )
    parser.add_argument("--target_col", type=int, default=0, help="the target column for separation")
    parser.add_argument("--n_iter", type=int, default=200, help="number of iteration")
    parser.add_argument("--g_eps", type=float, default=5e-10, help="minumum value used for initializing G_NM")
    parser.add_argument("--n_mic", type=int, default=4, help="number of microphone")
    parser.add_argument("--n_bit", type=int, default=64, help="number of microphone")
    parser.add_argument("--algo", type=str, default="IP", help="the method for updating Q")
    # parser.add_argument("--init_angle_deg", type=int,default=0)
    parser.add_argument("--init_angle_deg", type=int,default=110)
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
            print("Warning: cupy is not installed. 'gpu' argument should be set to -1. Switched to CPU./n")
            import numpy as xp

    import time

    separater = FastMNMF2(
        n_source=args.n_source,
        n_basis=args.n_basis,
        xp=xp,
        init_SCM=args.init_SCM,
        n_bit=args.n_bit,
        algo=args.algo,
        n_iter_init=args.n_iter_init,
        g_eps=args.g_eps,
        init_angle_deg=args.init_angle_deg,
        target_col=args.target_col,
        seed=400
    )

    # wav, sample_rate = sf.read(args.input_fname)
    # wav /= np.abs(wav).max() * 1.2
    # M = min(len(wav), args.n_mic)

    wav1, sample_rate = sf.read("R:/code/SoundSourceSeparation/src/separation/11_single.wav")
    wav2, sample_rate = sf.read("R:/code/SoundSourceSeparation/src/separation/12_single.wav")

    wav = wav1 + wav2
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
