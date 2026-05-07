import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift

def roundf(x):
    return np.floor(x + 0.5)

class Conf:
    # Пример конфигурации, которая должна быть определена извне,
    # воспроизводим структуру для того, чтобы класс NSST работал корректно.
    lpfilt = "maxflat"
    class sp:
        dcompSize = 4
        dcomp = [3, 3, 4, 4]
        dsize = [32, 32, 16, 16]

class NSST:
    def __init__(self):
        self.atrous_filters = []
        self.shearing_filters = []

    def atrous_filters_init(self):
        if Conf.lpfilt == "maxflat":
            h0 = np.array([
                [-7.900496718847182e-07, 0., 1.4220894093924927e-05, 2.5281589500310983e-05, -4.9773129328737247e-05, -0.00022753430550279883, -0.00033182086219158167],
                [0, 0, 0, 0, 0, 0, 0],
                [1.4220894093924927e-05, 0., -0.0002559760936906487, -0.00045506861100559767, 0.0008959163279172705, 0.004095617499050379, 0.00597277551944847],
                [2.5281589500310983e-05, 0., -0.00045506861100559767, 0.0009765625, 0.0015927401385195919, -0.0087890625, -0.01795090623402861],
                [-4.9773129328737247e-05, 0., 0.0008959163279172705, 0.0015927401385195919, -0.0031357071477104465, -0.014334661246676327, -0.020904714318069645],
                [-0.00022753430550279883, 0., 0.004095617499050379, -0.0087890625, -0.014334661246676327, 0.0791015625, 0.16155815610625748],
                [-0.00033182086219158167, 0., 0.00597277551944847, -0.01795090623402861, -0.020904714318069645, 0.16155815610625748, 0.3177420190660832]
            ], dtype=np.float32)

            g0 = np.array([
                [-6.391587676622346e-10, 0., 1.7257286726880333e-08, 3.067962084778726e-08, -1.3805829381504267e-07, -5.522331752601707e-07, -3.3747582932565985e-07, 1.9328161134105974e-06, 5.6949046198705095e-06, 7.649452131381623e-06],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1.7257286726880333e-08, 0., -4.65946741625769e-07, -8.283497628902559e-07, 3.727573933006152e-06, 1.4910295732024608e-05, 9.111847391792816e-06, -5.2186035062086126e-05, -0.00015376242473650378, -0.00020653520754730382],
                [3.067962084778726e-08, 0., -8.283497628902559e-07, -1.2809236054493144e-06, 6.6267981031220475e-06, 2.305662489808766e-05, 1.0064497559808503e-05, -8.06981871433068e-05, -0.00021814634152337594, -0.00028666046030363884],
                [-1.3805829381504267e-07, 0., 3.727573933006152e-06, 6.6267981031220475e-06, -2.9820591464049215e-05, -0.00011928236585619686, -7.289477913434253e-05, 0.000417488280496689, 0.0012300993978920302, 0.0016522816603784306],
                [-5.522331752601707e-07, 0., 1.4910295732024608e-05, 2.305662489808766e-05, -0.00011928236585619686, -0.00041501924816557786, -0.00018116095607655303, 0.0014525673685795225, 0.0039266341474207675, 0.005159888285465499],
                [-3.3747582932565985e-07, 0., 9.111847391792816e-06, 1.0064497559808503e-05, -7.289477913434253e-05, -0.00018116095607655303, 0.001468581806076247, 0.0006340633462679356, -0.01181401175635013, -0.021745034491193898],
                [1.9328161134105974e-06, 0., -5.2186035062086126e-05, -8.06981871433068e-05, 0.000417488280496689, 0.0014525673685795225, 0.0006340633462679356, -0.005083985790028328, -0.013743219515972684, -0.018059608999129246],
                [5.6949046198705095e-06, 0., -0.00015376242473650378, -0.00021814634152337594, 0.0012300993978920302, 0.0039266341474207675, -0.01181401175635013, -0.013743219515972684, 0.0826466923977296, 0.1638988884584603],
                [7.649452131381623e-06, 0., -0.00020653520754730382, -0.00028666046030363884, 0.0016522816603784306, 0.005159888285465499, -0.021745034491193898, -0.018059608999129246, 0.1638988884584603, 0.31358726209239235]
            ], dtype=np.float32)

            g1 = np.array([
                [-7.900496718847182e-07, 0., 1.4220894093924927e-05, 2.5281589500310983e-05, -4.9773129328737247e-05, -0.00022753430550279883, -0.00033182086219158167],
                [0, 0, 0, 0, 0, 0, 0],
                [1.4220894093924927e-05, 0., -0.0002559760936906487, -0.00045506861100559767, 0.0008959163279172705, 0.004095617499050379, 0.00597277551944847],
                [2.5281589500310983e-05, 0., -0.00045506861100559767, -0.0009765625, 0.0015927401385195919, 0.0087890625, 0.01329909376597139],
                [-4.9773129328737247e-05, 0., 0.0008959163279172705, 0.0015927401385195919, -0.0031357071477104465, -0.014334661246676327, -0.020904714318069645],
                [-0.00022753430550279883, 0., 0.004095617499050379, 0.0087890625, -0.014334661246676327, -0.0791015625, -0.1196918438937425],
                [-0.00033182086219158167, 0., 0.00597277551944847, 0.01329909376597139, -0.020904714318069645, -0.1196918438937425, 0.8177420190660831]
            ], dtype=np.float32)

            h1 = np.array([
                [6.391587676622346e-10, 0.0, -1.7257286726880333e-08, -3.067962084778726e-08, 1.3805829381504267e-07, 5.522331752601707e-07, 3.3747582932565985e-07, -1.9328161134105974e-06, -5.6949046198705095e-06, -7.649452131381623e-06],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.7257286726880333e-08, 0.0, 4.65946741625769e-07, 8.283497628902559e-07, -3.727573933006152e-06, -1.4910295732024608e-05, -9.111847391792816e-06, 5.2186035062086126e-05, 0.00015376242473650378, 0.00020653520754730382],
                [-3.067962084778726e-08, 0.0, 8.283497628902559e-07, -2.9917573832012203e-07, -6.6267981031220475e-06, 5.3851632897621965e-06, 4.049868144081346e-05, -1.884807151416769e-05, -0.00023692226948222173, -0.0003769812640795245],
                [1.3805829381504267e-07, 0.0, -3.727573933006152e-06, -6.6267981031220475e-06, 2.9820591464049215e-05, 0.00011928236585619686, 7.289477913434253e-05, -0.000417488280496689, -0.0012300993978920302, -0.0016522816603784306],
                [5.522331752601707e-07, 0.0, -1.4910295732024608e-05, 5.3851632897621965e-06, 0.00011928236585619686, -9.693293921571956e-05, -0.0007289762659346422, 0.00033926528725501844, 0.004264600850679991, 0.006785662753431441],
                [3.3747582932565985e-07, 0.0, -9.111847391792816e-06, 4.049868144081346e-05, 7.289477913434253e-05, -0.0007289762659346422, -0.001468581806076247, 0.002551416930771248, 0.01181401175635013, 0.017093222023136675],
                [-1.9328161134105974e-06, 0.0, 5.2186035062086126e-05, -1.884807151416769e-05, -0.000417488280496689, 0.00033926528725501844, 0.002551416930771248, -0.0011874285053925643, -0.01492610297737997, -0.023749819637010044],
                [-5.6949046198705095e-06, 0.0, 0.00015376242473650378, -0.00023692226948222173, -0.0012300993978920302, 0.004264600850679991, 0.01181401175635013, -0.01492610297737997, -0.0826466923977296, -0.12203257624594532],
                [-7.649452131381623e-06, 0.0, 0.00020653520754730382, -0.0003769812640795245, -0.0016522816603784306, 0.006785662753431441, 0.017093222023136675, -0.023749819637010044, -0.12203257624594532, 0.821896776039774]
            ], dtype=np.float32)

            def make_filter(f):
                h_ext = np.concatenate([f, np.fliplr(f[:, :-1])], axis=1)
                return np.concatenate([h_ext, np.flipud(h_ext[:-1, :])], axis=0)

            self.atrous_filters = [
                make_filter(g0),
                make_filter(h0),
                make_filter(g1),
                make_filter(h1)
            ]
        else:
            raise ValueError("Only 'maxflat' is supported in this implementation")

    def avg_pol(self, L, x1, y1, x2, y2):
        d = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            for j in range(L):
                d[int(y1[i, j] - 1), int(x1[i, j] - 1)] += 1
                d[int(y2[i, j] - 1), int(x2[i, j] - 1)] += 1
        return d

    def gen_xy_coordinates(self, n):
        n += 1
        x1, y1, x2, y2 = [np.zeros((n, n), dtype=np.float32) for _ in range(4)]
        y0 = 1

        for i in range(n):
            x0_v = i + 1
            xN = n - x0_v + 1
            if xN == x0_v:
                flag = 1
            else:
                m = (n - y0) / (xN - x0_v)
                flag = 0

            xt = np.linspace(x0_v, xN, n)

            for j in range(n):
                if flag == 0:
                    x1[i, j] = y2[i, j] = roundf(xt[j])
                    x2[i, j] = y1[i, j] = roundf(m * (xt[j] - x0_v) + y0)
                else:
                    x1[i, j] = y2[i, j] = int((n - 1) / 2) + 1
                    x2[i, j] = y1[i, j] = j + 1

        n -= 1
        x1 = np.fliplr(x1[:n, :n])
        y1 = y1[:n, :n]
        x2 = x2[1:n+1, :n]
        y2 = y2[1:n+1, :n]
        y2[n-1, 0] = n

        D = self.avg_pol(n, x1, y1, x2, y2)
        return [x1, y1, x2, y2, D]

    def meyer_wind(self, x):
        y = 0.0
        x_less = -1/3 + 1/2
        x_greater = 1/3 + 1/2
        x_le1 = 1/3 + 1/2; x_ge1 = 2/3 + 1/2
        x_le2 = -2/3 + 1/2; x_ge2 = 1/3 + 1/2

        if x_less < x < x_greater:
            y = 1.0
        elif (x_le1 <= x <= x_ge1) or (x_le2 <= x <= x_ge2):
            w = 3.0 * abs(x - 1/2) - 1.0
            z = (w**4) * (35.0 - 84.0*w + 70.0*(w**2) - 20.0*(w**3))
            y = (np.cos(np.pi/2 * z))**2
        return y

    def windowing(self, x_in, N, L):
        y = np.zeros((N, L), dtype=np.float32)
        T = N / L
        n = int(2.0 * T)
        g = np.array([self.meyer_wind(j / T - 0.5) for j in range(n)])

        for j in range(L):
            index = 0
            for k in range(int(-T/2), int(1.5*T)):
                in_sig = int(np.floor((k + j*T) % N))
                # Protect out of bounds index conceptually
                if index < len(g):
                    y[in_sig, j] = g[index] * x_in[in_sig]
                index += 1
        return y

    def rec_from_pol(self, l_arr, n, gen):
        x1, y1, x2, y2, D = gen
        matC = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                matC[int(y1[i, j]-1), int(x1[i, j]-1)] += l_arr[i, j]
                matC[int(y2[i, j]-1), int(x2[i, j]-1)] += l_arr[i+n, j]
        return matC / np.where(D == 0, 1, D)

    def shearing_filters_myer(self):
        self.shearing_filters = []
        for t in range(Conf.sp.dcompSize):
            n_size = Conf.sp.dsize[t]
            gen = self.gen_xy_coordinates(n_size)
            ones_ = np.ones(2 * n_size, dtype=np.float32)
            L = int(2 ** Conf.sp.dcomp[t])
            wf = self.windowing(ones_, 2 * n_size, L)
            
            level_filters = []
            for i in range(L):
                temp = wf[:, i].reshape(-1, 1) @ np.ones((1, n_size), dtype=np.float32)
                f_polar = self.rec_from_pol(temp, n_size, gen)
                f_shifted = fftshift(f_polar)
                f_ifft = np.real(fftshift(ifft2(f_shifted)))
                level_filters.append(f_ifft / np.sqrt(n_size))
            
            self.shearing_filters.append(level_filters)

    def symext(self, x, h, shift):
        m, n = x.shape
        p, q = h.shape
        s1, s2 = int(shift[0]), int(shift[1])

        ss = int(np.floor(p / 2)) - s1 + 1
        rr = int(np.floor(q / 2)) - s2 + 1

        extended_fliplr = np.concatenate([np.fliplr(x[:, :ss]), x], axis=1)
        yT = np.concatenate([extended_fliplr, np.fliplr(x[:, n-(p+s1):])], axis=1)

        extended_flipud = np.concatenate([np.flipud(yT[:rr, :]), yT], axis=0)
        yT2 = np.concatenate([extended_flipud, yT[m-(q+s2):, :]], axis=0)

        return yT2[:m+p-1, :n+q-1]

    def atrousc(self, signal, filt, upsampling_factor):
        M0 = upsampling_factor
        sM0 = M0 - 1
        SRowLength, SColLength = signal.shape
        FRowLength, FColLength = filt.shape
        O_SRowLength = SRowLength - M0 * FRowLength + 1
        O_SColLength = SColLength - M0 * FColLength + 1
        filt_flipped = filt[::-1, ::-1]
        # vectorized: build (FRow, O_Row, FCol, O_Col) patch tensor, contract with filter
        rows = np.arange(FRowLength) * M0
        cols = np.arange(FColLength) * M0
        patches = signal[
            sM0 + rows[:, None, None, None] + np.arange(O_SRowLength)[None, :, None, None],
            sM0 + cols[None, None, :, None] + np.arange(O_SColLength)[None, None, None, :]
        ]  # shape: (FRow, O_Row, FCol, O_Col)
        # filt_flipped: (FRow, FCol) = (a, c); patches: (a, b, c, d); out: (b, d)
        return np.einsum('ac,abcd->bd', filt_flipped, patches).astype(np.float32)

    def upsample2df(self, filt, level):
        M0 = int(2 ** level)
        new_h = filt.shape[0] * M0
        new_w = filt.shape[1] * M0
        upsampled = np.zeros((new_h, new_w), dtype=np.float32)
        upsampled[::M0, ::M0] = filt
        return upsampled

    def atrous_dec(self, image, level):
        y0_filt = self.atrous_filters[1]
        y1_filt = self.atrous_filters[3]
        
        y = [None] * (level + 1)
        
        shift = [1.0, 1.0]
        sym = self.symext(image, y0_filt, shift)
        img_low = convolve2d(sym, y0_filt, mode='valid')
        
        sym = self.symext(image, y1_filt, shift)
        img_high = convolve2d(sym, y1_filt, mode='valid')
        
        y[level] = img_high
        curr_image = img_low

        for i in range(1, level):
            shift = [2.0 - 2**(i-1), 2.0 - 2**(i-1)]
            upsampling_factor = 2**i
            
            sym = self.symext(curr_image, self.upsample2df(y0_filt, i), shift)
            img_low = self.atrousc(sym, y0_filt, upsampling_factor)
            
            sym = self.symext(curr_image, self.upsample2df(y1_filt, i), shift)
            img_high = self.atrousc(sym, y1_filt, upsampling_factor)
            
            y[level - i] = img_high
            curr_image = img_low

        y[0] = curr_image
        return y

    def dec(self, image):
        level = Conf.sp.dcompSize
        y_atrous = self.atrous_dec(image, level)
        
        dst = [y_atrous[0]]
        
        for i in range(level):
            size = int(2 ** Conf.sp.dcomp[i])
            level_dst = []
            for k in range(size):
                shear_f = self.shearing_filters[i][k] * np.sqrt(Conf.sp.dsize[i])
                temp = convolve2d(y_atrous[i+1], shear_f, mode='same', boundary='wrap')
                temp = np.roll(temp, shift=(-1, -1), axis=(0, 1))
                level_dst.append(temp)
            dst.append(level_dst)
            
        return dst

    def atrous_rec(self, y):
        NLevels = len(y) - 1
        x = np.copy(y[0])
        
        g0_filt = self.atrous_filters[0]
        g1_filt = self.atrous_filters[2]
        
        temp_p = x
        for i in range(NLevels - 1, 0, -1):
            shift = [2.0 - 2**(i-1), 2.0 - 2**(i-1)]
            upsampling_factor = 2**i
            
            s1 = self.symext(temp_p, self.upsample2df(g0_filt, i), shift)
            c1 = self.atrousc(s1, g0_filt, upsampling_factor)
            
            s2 = self.symext(y[NLevels - i], self.upsample2df(g1_filt, i), shift)
            c2 = self.atrousc(s2, g1_filt, upsampling_factor)
            
            temp_p = c1 + c2
            
        shift = [1.0, 1.0]
        s1 = self.symext(temp_p, g0_filt, shift)
        c1 = convolve2d(s1, g0_filt, mode='valid')
        
        s2 = self.symext(y[NLevels], g1_filt, shift)
        c2 = convolve2d(s2, g1_filt, mode='valid')
        
        return c1 + c2

    def list_sum_axis0(self, target_list):
        return np.sum(target_list, axis=0)

    def rec(self, dst):
        level = len(dst) - 1
        y = [dst[0]]
        
        for i in range(1, level + 1):
            # В C++ называется Sum_Axis3 (складывает все направления одного уровня)
            y.append(self.list_sum_axis0(dst[i]))
            
        return self.atrous_rec(y)

# Пример использования
if __name__ == "__main__":
    nsst = NSST()
    nsst.atrous_filters_init()
    nsst.shearing_filters_myer()
    
    # Minimum safe size: atrous level-3 uses upsampled h1 (19x19 * 8 = 152), needs image >= 150px
    img = np.random.randn(256, 256).astype(np.float32)
    
    # Разложение
    coeffs = nsst.dec(img)
    
    # Восстановление
    reconstructed = nsst.rec(coeffs)
    print("Reconstructed shape:", reconstructed.shape)
    print("Max abs error:", np.abs(img - reconstructed).max())