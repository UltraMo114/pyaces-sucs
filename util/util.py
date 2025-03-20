import math
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

class ColorOperations:
    GAMUT_TABLE_SIZE = 360
    ADDITIONAL_TABLE_ENTRIES = 2
    TOTAL_TABLE_SIZE = GAMUT_TABLE_SIZE + ADDITIONAL_TABLE_ENTRIES
    BASE_INDEX = 1

    # --- Data Classes ---
    @dataclass
    class Chromaticities:
        """RGB Chromaticity coordinates"""
        red: List[float]    # [x, y]
        green: List[float]  # [x, y]
        blue: List[float]   # [x, y]
        white: List[float]  # [x, y]

    @dataclass
    class ODTParams:
        """Output Display Transform Parameters"""
        reference_luminance: float
        peak_luminance: float
        n_r: float
        g: float
        t_1: float
        c_t: float
        s_2: float
        u_2: float
        m_2: float
        limit_jmax: float
        mid_j: float
        model_gamma: float
        sat: float
        sat_thr: float
        compr: float
        chroma_compress_scale: float
        focus_dist: float
        limit_rgb_to_xyz: np.ndarray
        limit_xyz_to_rgb: np.ndarray
        xyz_w_limit: np.ndarray
        output_rgb_to_xyz: np.ndarray
        output_xyz_to_rgb: np.ndarray
        xyz_w_output: np.ndarray
        lower_hull_gamma: float

    @dataclass
    class TSParams:
        """Tonescale Parameters"""
        n: float
        n_r: float
        g: float
        t_1: float
        c_t: float
        s_2: float
        u_2: float
        m_2: float

    # --- Initialization Methods ---
    def __init__(self,reference_luminance:float, peak_luminance: float, limiting_primaries: Chromaticities,
                 encoding_primaries: Chromaticities, viewing_conditions: int = 1):
        self.input_primaries = self.Chromaticities(
            red=[0.73470, 0.26530],
            green=[0.00000, 1.00000],
            blue=[0.00010, -0.07700],
            white=[0.32168, 0.33767]
        )
        self.odt_params = self._init_odt_params(reference_luminance, peak_luminance, limiting_primaries, encoding_primaries, viewing_conditions)
        self.ts_params = self._init_ts_params(peak_luminance)
        self.reachm_table = self.make_reachm_table(limiting_primaries, self.odt_params.limit_jmax, peak_luminance)
        self.gamut_cusp_table = self.make_gamut_table(encoding_primaries, peak_luminance)
        self.gamut_top_gamma = np.array([1.14] * self.TOTAL_TABLE_SIZE)
    def _init_odt_params(self,reference_luminance: float, peak_luminance: float, limiting_primaries: Chromaticities,
                         encoding_primaries: Chromaticities, viewing_conditions: int) -> 'ODTParams':
        Y_B = 20.0
        L_A = 100.0
        FOCUS_DISTANCE = 1.35
        FOCUS_DISTANCE_SCALING = 1.75

        ts_params = self._init_ts_params(peak_luminance)
        limit_jmax = self._y_to_sucs_j(peak_luminance)
        mid_j = self._y_to_sucs_j(ts_params.c_t * 100.0)

        chroma_compress = 2.4
        chroma_compress_fact = 3.3
        chroma_expand = 1.3
        chroma_expand_fact = 0.69
        chroma_expand_thr = 0.5

        log_peak = math.log10(ts_params.n / ts_params.n_r)
        compr = chroma_compress + (chroma_compress * chroma_compress_fact) * log_peak
        sat = max(0.2, chroma_expand - (chroma_expand * chroma_expand_fact) * log_peak)
        sat_thr = chroma_expand_thr / ts_params.n
        chroma_compress_scale = math.pow(0.03379 * ts_params.n, 0.30596) - 0.45135

        surround = self._viewing_conditions_to_surround(viewing_conditions)
        model_gamma = 1.0 / (surround[1] * (1.48 + math.sqrt(Y_B / L_A)))
        focus_dist = FOCUS_DISTANCE + FOCUS_DISTANCE * FOCUS_DISTANCE_SCALING * log_peak

        rgb_w = np.array([100.0, 100.0, 100.0])
        limit_rgb_to_xyz = self._rgb_to_xyz_f33(limiting_primaries, 1.0)
        limit_xyz_to_rgb = self._xyz_to_rgb_f33(limiting_primaries, 1.0)
        xyz_w_limit = np.dot(rgb_w, limit_rgb_to_xyz.T)
        output_rgb_to_xyz = self._rgb_to_xyz_f33(encoding_primaries, 1.0)
        output_xyz_to_rgb = self._xyz_to_rgb_f33(encoding_primaries, 1.0)
        xyz_w_output = np.dot(rgb_w, output_rgb_to_xyz.T)

        return self.ODTParams(
            reference_luminance=reference_luminance,
            peak_luminance=peak_luminance,
            n_r=ts_params.n_r,
            g=ts_params.g,
            t_1=ts_params.t_1,
            c_t=ts_params.c_t,
            s_2=ts_params.s_2,
            u_2=ts_params.u_2,
            m_2=ts_params.m_2,
            limit_jmax=limit_jmax,
            mid_j=mid_j,
            model_gamma=model_gamma,
            sat=sat,
            sat_thr=sat_thr,
            compr=compr,
            chroma_compress_scale=chroma_compress_scale,
            focus_dist=focus_dist,
            limit_rgb_to_xyz=limit_rgb_to_xyz,
            limit_xyz_to_rgb=limit_xyz_to_rgb,
            xyz_w_limit=xyz_w_limit,
            output_rgb_to_xyz=output_rgb_to_xyz,
            output_xyz_to_rgb=output_xyz_to_rgb,
            xyz_w_output=xyz_w_output,
            lower_hull_gamma=1.14 + 0.07 * log_peak
        )

    def _init_ts_params(self, peak_luminance: float) -> 'TSParams':
        n = peak_luminance
        n_r = 100.0
        g = 1.15
        c = 0.18
        c_d = 10.013
        w_g = 0.14
        t_1 = 0.04
        r_hit_min = 128.0
        r_hit_max = 896.0

        r_hit = r_hit_min + (r_hit_max - r_hit_min) * (math.log(n / n_r) / math.log(10000.0 / 100.0))
        m_0 = n / n_r
        m_1 = 0.5 * (m_0 + math.sqrt(m_0 * (m_0 + 4.0 * t_1)))
        u = math.pow((r_hit / m_1) / ((r_hit / m_1) + 1), g)
        m = m_1 / u
        w_i = math.log(n / 100.0) / math.log(2.0)
        c_t = c_d / n_r * (1.0 + w_i * w_g)
        g_ip = 0.5 * (c_t + math.sqrt(c_t * (c_t + 4.0 * t_1)))
        g_ipp2 = -(m_1 * math.pow((g_ip / m), (1.0 / g))) / (math.pow(g_ip / m, 1.0 / g) - 1.0)
        w_2 = c / g_ipp2
        s_2 = w_2 * m_1
        u_2 = math.pow((r_hit / m_1) / ((r_hit / m_1) + w_2), g)
        m_2 = m_1 / u_2

        return self.TSParams(n, n_r, g, t_1, c_t, s_2, u_2, m_2)
    
    def xyz_to_jch(self, xyz: List[float], xyzw: List[float]) -> List[float]:
        """Convert XYZ to JCh color space."""
        xyz_to_lms = np.array([[0.4002, 0.7075, -0.0807], [-0.2280, 1.1500, 0.0612], [0.0000, 0.0000, 0.9184]])
        lms_to_iab = np.array([[200/3.05, 100/3.05, 5/3.05], [430.0, -470.0, 40.0], [49.0, 49.0, -98.0]])
        xyz_scaled = np.array([xyz[0] / xyzw[0] * 0.95047, xyz[1] / xyzw[1] * 1.0, xyz[2] / xyzw[2] * 1.0883])
        lms = np.dot(xyz_to_lms, xyz_scaled)
        lms_trans = np.sign(lms) * np.power(np.abs(lms), 0.43)
        iab = np.dot(lms_to_iab, lms_trans)
        c_part = math.sqrt(max(1e-9, iab[1]**2 + iab[2]**2))
        a_part = iab[1] / c_part if c_part > 1e-5 else 0.0
        b_part = iab[2] / c_part if c_part > 1e-5 else 0.0
        c = (1 / 0.0252) * math.log1p(0.0447 * c_part)
        j = np.clip(iab[0], 1e-5, 100.0)
        h = math.atan2(b_part, a_part) * (180 / math.pi)
        if h < 0:
            h += 360
        return [j, c, h]
    
    def rgb_to_jch(self, rgb: List[float], rgb_to_xyz_m: np.ndarray, peak_luminance: float) -> List[float]:
        """Convert RGB to JCh color space."""
        rgbw = [self.odt_params.reference_luminance, self.odt_params.reference_luminance, self.odt_params.reference_luminance]
        luminance_rgb = np.array(rgb) * peak_luminance
        xyzw = np.dot(rgbw, rgb_to_xyz_m.T)
        luminance_xyz = np.dot(luminance_rgb, rgb_to_xyz_m.T)
        return self.xyz_to_jch(luminance_xyz, xyzw)
    
    def rgb_to_jch_vectorized(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB to JCh color space with vectorized operations.
        
        Parameters:
        -----------
        rgb : np.ndarray
            RGB values with shape (-1, 3)
        rgb_to_xyz_m : np.ndarray
            RGB to XYZ conversion matrix
        peak_luminance : float
            Peak luminance value
        reference_luminance : float
            Reference luminance value

        Returns:
        --------
        np.ndarray
            JCh values with same shape as input
        """
        rgb_to_xyz_m = self.odt_params.limit_rgb_to_xyz
        # Constants
        xyz_to_lms = np.array([[0.4002, 0.7075, -0.0807], 
                            [-0.2280, 1.1500, 0.0612], 
                            [0.0000, 0.0000, 0.9184]])
        
        lms_to_iab = np.array([[200/3.05, 100/3.05, 5/3.05], 
                            [430.0, -470.0, 40.0], 
                            [49.0, 49.0, -98.0]])
        
        # Calculate white point
        rgbw = np.array([self.odt_params.reference_luminance, self.odt_params.reference_luminance, self.odt_params.reference_luminance])
        xyzw = np.dot(rgbw, rgb_to_xyz_m.T)
        
        # Convert RGB to XYZ
        batch_size = rgb.shape[0] if len(rgb.shape) > 1 else 1
        rgb_reshaped = rgb.reshape(-1, 3)
        luminance_rgb = rgb_reshaped * self.odt_params.peak_luminance
        luminance_xyz = np.dot(luminance_rgb, rgb_to_xyz_m.T)
        
        # Scale XYZ by white point
        xyz_scaled = np.zeros_like(luminance_xyz)
        xyz_scaled[:, 0] = luminance_xyz[:, 0] / xyzw[0] * 0.95047
        xyz_scaled[:, 1] = luminance_xyz[:, 1] / xyzw[1] * 1.0
        xyz_scaled[:, 2] = luminance_xyz[:, 2] / xyzw[2] * 1.0883
        
        # Convert to LMS
        lms = np.dot(xyz_scaled, xyz_to_lms.T)
        
        # Apply power function to LMS
        lms_trans = np.sign(lms) * np.power(np.abs(lms), 0.43)
        
        # Convert to IAB
        iab = np.dot(lms_trans, lms_to_iab.T)
        
        # Calculate JCh
        j = np.clip(iab[:, 0], 1e-5, 100.0)
        
        # Calculate C (chroma)
        c_part = np.sqrt(np.maximum(1e-9, iab[:, 1]**2 + iab[:, 2]**2))
        c = (1 / 0.0252) * np.log1p(0.0447 * c_part)
        
        # Calculate h (hue)
        h = np.arctan2(iab[:, 2], iab[:, 1]) * (180 / np.pi)
        h = np.where(h < 0, h + 360, h)
        
        # Combine results
        jch = np.column_stack((j, c, h))
        
        # Return in original shape
        if len(rgb.shape) == 1:
            return jch[0]
        return jch

    def jch_to_rgb_vectorized(self, jch: np.ndarray) -> np.ndarray:
        """
        Convert JCh to RGB color space with vectorized operations.
        
        Parameters:
        -----------
        jch : np.ndarray
            JCh values with shape (-1, 3)
        xyz_to_rgb_m : np.ndarray
            XYZ to RGB conversion matrix
            
        Returns:
        --------
        np.ndarray
            RGB values with same shape as input
        """
        xyz_to_rgb_m = self.odt_params.output_xyz_to_rgb
        # Constants
        iab_to_lms = np.array([[0.01, 0.00074681, 0.0004721], 
                            [0.01, -0.00147541, -0.00043493], 
                            [0.01, -0.0003643, -0.01018549]])
        
        lms_to_xyz = np.array([[1.8502, -1.1383, 0.2384], 
                            [0.3668, 0.6439, -0.0107], 
                            [0.0000, 0.0000, 1.0889]])
        
        # Calculate white point
        rgbw = np.array([self.odt_params.reference_luminance, 
                        self.odt_params.reference_luminance, 
                        self.odt_params.reference_luminance])
        
        rgb_to_xyz = np.linalg.inv(xyz_to_rgb_m)
        xyzw = np.dot(rgbw, rgb_to_xyz.T)
        
        # Reshape input if needed
        original_shape = jch.shape
        jch_reshaped = jch.reshape(-1, 3)
        
        # Convert h to radians
        h_rad = jch_reshaped[:, 2] * (np.pi / 180.0)
        
        # Calculate c_part
        c_part = (np.exp(0.0252 * jch_reshaped[:, 1]) - 1) / 0.0447
        
        # Create IAB array
        iab = np.zeros_like(jch_reshaped)
        iab[:, 0] = jch_reshaped[:, 0]
        iab[:, 1] = c_part * np.cos(h_rad)
        iab[:, 2] = c_part * np.sin(h_rad)
        
        # Convert to LMS
        lms_trans = np.dot(iab, iab_to_lms.T)
        lms = np.sign(lms_trans) * np.power(np.abs(lms_trans), 1/0.43)
        
        # Convert to XYZ
        xyz = np.dot(lms, lms_to_xyz.T)
        
        # Scale by white point
        luminance_xyz = np.zeros_like(xyz)
        luminance_xyz[:, 0] = xyz[:, 0] * xyzw[0] / 0.95047
        luminance_xyz[:, 1] = xyz[:, 1] * xyzw[1]
        luminance_xyz[:, 2] = xyz[:, 2] * xyzw[2] / 1.0883
        
        # Convert to RGB
        luminance_rgb = np.dot(luminance_xyz, xyz_to_rgb_m.T)
        return luminance_rgb

    
    def jch_to_xyz(self, jch: List[float], xyzw: List[float]) -> List[float]:
        """Convert JCh to XYZ color space."""
        iab_to_lms = np.array([[0.01, 0.00074681, 0.0004721], [0.01, -0.00147541, -0.00043493], [0.01, -0.0003643, -0.01018549]])
        lms_to_xyz = np.array([[1.8502, -1.1383, 0.2384], [0.3668, 0.6439, -0.0107], [0.0000, 0.0000, 1.0889]])
        h_rad = jch[2] * (math.pi / 180.0)
        c_part = (math.exp(0.0252 * jch[1]) - 1) / 0.0447
        iab = np.array([jch[0], c_part * math.cos(h_rad), c_part * math.sin(h_rad)])
        lms_trans = np.dot(iab_to_lms, iab)
        lms = np.sign(lms_trans) * np.power(np.abs(lms_trans), 1/0.43)
        xyz = np.dot(lms_to_xyz, lms)
        return [xyz[0] * xyzw[0] / 0.95047, xyz[1] * xyzw[1], xyz[2] * xyzw[2] / 1.0883]
    
    def jch_to_rgb(self, jch: List[float], xyz_to_rgb_m: np.ndarray, peak_luminance: float) -> List[float]:
        """Convert JCh to RGB color space."""
        rgbw = [self.odt_params.reference_luminance, self.odt_params.reference_luminance, self.odt_params.reference_luminance]
        rgb_to_xyz = np.linalg.inv(xyz_to_rgb_m)
        xyzw = np.dot(rgbw, rgb_to_xyz.T)
        luminance_xyz = self.jch_to_xyz(jch, xyzw)
        luminance_rgb = np.dot(luminance_xyz, xyz_to_rgb_m.T)
        return luminance_rgb / peak_luminance
    
    def make_gamut_table(self, c: Chromaticities, peak_luminance: float) -> np.ndarray:
        """Create a gamut boundary table for gamut mapping."""
        rgb_to_xyz_m = self._rgb_to_xyz_f33(c, 1.0)
        gamut_cusp_table_unsorted = np.zeros((self.GAMUT_TABLE_SIZE, 3))
        for i in range(self.GAMUT_TABLE_SIZE):
            h_norm = float(i) / self.GAMUT_TABLE_SIZE
            hsv = [h_norm, 1.0, 1.0]
            rgb = self.hsv_to_rgb(hsv)
            gamut_cusp_table_unsorted[i] = self.rgb_to_jch(rgb, rgb_to_xyz_m, peak_luminance)
        min_h_index = np.argmin(gamut_cusp_table_unsorted[:, 2])
        gamut_cusp_table = np.zeros((self.TOTAL_TABLE_SIZE, 3))
        for i in range(self.GAMUT_TABLE_SIZE):
            idx = (min_h_index + i) % self.GAMUT_TABLE_SIZE
            gamut_cusp_table[i + self.BASE_INDEX] = gamut_cusp_table_unsorted[idx]
        gamut_cusp_table[0] = gamut_cusp_table[self.BASE_INDEX + self.GAMUT_TABLE_SIZE - 1]
        gamut_cusp_table[self.BASE_INDEX + self.GAMUT_TABLE_SIZE] = gamut_cusp_table[self.BASE_INDEX]
        gamut_cusp_table[0, 2] -= 360.0
        gamut_cusp_table[self.GAMUT_TABLE_SIZE + 1, 2] += 360.0
        return gamut_cusp_table

    def make_reachm_table(self, c: Chromaticities, limit_jmax: float, peak_luminance: float) -> np.ndarray:
        """Create a table of reach M values at limit_jmax."""
        xyz_to_rgb_m = self._xyz_to_rgb_f33(c, 1.0)
        reach_table = [0.0] * self.GAMUT_TABLE_SIZE
        for i in range(self.GAMUT_TABLE_SIZE):
            hue = i * 360.0 / self.GAMUT_TABLE_SIZE
            low = 0.0
            high = 50.0
            while high < 1300.0:
                search_jmh = [limit_jmax, high, hue]
                new_limit_rgb = self.jch_to_rgb(search_jmh, xyz_to_rgb_m, peak_luminance)
                if any(x < 0.0 for x in new_limit_rgb):
                    break
                low = high
                high += 50.0
            while high - low > 1e-1:
                sample_m = (high + low) / 2.0
                search_jmh = [limit_jmax, sample_m, hue]
                new_limit_rgb = self.jch_to_rgb(search_jmh, xyz_to_rgb_m, peak_luminance)
                if any(x < 0.0 for x in new_limit_rgb):
                    high = sample_m
                else:
                    low = sample_m
            reach_table[i] = high
        return np.array(reach_table)
    
    def _rgb_to_xyz_f33(self, primaries: Chromaticities, luminance: float) -> np.ndarray:
        rx, ry = primaries.red
        gx, gy = primaries.green
        bx, by = primaries.blue
        wx, wy = primaries.white
        rym = ry / wy
        gym = gy / wy
        bym = by / wy
        Xr, Yr, Zr = rx * rym, ry * rym, (1 - rx - ry) * rym
        Xg, Yg, Zg = gx * gym, gy * gym, (1 - gx - gy) * gym
        Xb, Yb, Zb = bx * bym, by * bym, (1 - bx - by) * bym
        Xw, Yw, Zw = wx / wy, wy / wy, (1 - wx - wy) / wy
        s = np.linalg.solve(np.array([[Xr, Xg, Xb], [Yr, Yg, Yb], [Zr, Zg, Zb]]), np.array([Xw, Yw, Zw]))
        Sr, Sg, Sb = s
        return np.array([[Sr * Xr, Sg * Xg, Sb * Xb], [Sr * Yr, Sg * Yg, Sb * Yb], [Sr * Zr, Sg * Zg, Sb * Zb]])

    def _xyz_to_rgb_f33(self, primaries: Chromaticities, luminance: float) -> np.ndarray:
        return np.linalg.inv(self._rgb_to_xyz_f33(primaries, luminance))
    
    def _viewing_conditions_to_surround(self, viewing_conditions: int) -> List[float]:
        if viewing_conditions == 0:
            return [0.8, 0.52]
        elif viewing_conditions == 2:
            return [1.0, 0.69]
        return [0.9, 0.605]
    
    def hsv_to_rgb(self, hsv: List[float]) -> List[float]:
        c = hsv[2] * hsv[1]
        x = c * (1.0 - abs(math.fmod(hsv[0] * 6.0, 2.0) - 1.0))
        m = hsv[2] - c
        if hsv[0] < 1.0/6.0:
            rgb = [c, x, 0.0]
        elif hsv[0] < 2.0/6.0:
            rgb = [x, c, 0.0]
        elif hsv[0] < 3.0/6.0:
            rgb = [0.0, c, x]
        elif hsv[0] < 4.0/6.0:
            rgb = [0.0, x, c]
        elif hsv[0] < 5.0/6.0:
            rgb = [x, 0.0, c]
        else:
            rgb = [c, 0.0, x]
        return [r + m for r in rgb]
    
    # Vectorized functions
    def chroma_compression(self, jmh: np.ndarray, orig_j: np.ndarray, reach_m_table: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Compress chroma to adjust colorfulness.
        
        Args:
            jmh: Array of shape (-1, 3) containing JMh values
            orig_j: Array of shape (-1,) containing original J values
            reach_m_table: Array containing reach M table values
            invert: Whether to invert the compression
            
        Returns:
            Compressed M values as array of shape (-1,)
        """
        j = jmh[:, 0]
        m = jmh[:, 1]
        h = jmh[:, 2]
        
        # Early return for zero chroma
        zero_mask = (m == 0.0)
        
        n_j = j / self.odt_params.limit_jmax
        sn_j = np.maximum(0.0, 1.0 - n_j)
        m_norm = self.chroma_compression_norm(h)
        limit = np.power(n_j, self.odt_params.model_gamma) * self.reach_m_from_table(h, reach_m_table) / m_norm
        toe_limit = limit - 0.001
        toe_sn_j_sat = sn_j * self.odt_params.sat
        toe_sqrt_n_j_sat_thr = np.sqrt(n_j * n_j + self.odt_params.sat_thr)
        toe_n_j_compr = n_j * self.odt_params.compr
        
        result = np.zeros_like(m)
        
        if not invert:
            result = m * np.power(j / orig_j, self.odt_params.model_gamma) / m_norm
            result = limit - self.toe(limit - result, toe_limit, toe_sn_j_sat, toe_sqrt_n_j_sat_thr, False)
            result = self.toe(result, limit, toe_n_j_compr, sn_j, False)
            result = result * m_norm
        else:
            result = m / m_norm
            result = self.toe(result, limit, toe_n_j_compr, sn_j, True)
            result = limit - self.toe(limit - result, toe_limit, toe_sn_j_sat, toe_sqrt_n_j_sat_thr, True)
            result = result * m_norm * np.power(j / orig_j, -self.odt_params.model_gamma)
        
        # Handle zero chroma case
        result[zero_mask] = 0.0
        
        return result
    
    def chroma_compression_norm(self, h: np.ndarray) -> np.ndarray:
        """
        Vectorized version of chroma_compression_norm
        
        Args:
            h: Array of hue values
            
        Returns:
            Normalized chroma compression values
        """
        hr = np.radians(h)
        a = np.cos(hr)
        b = np.sin(hr)
        cos_hr2 = a * a - b * b
        sin_hr2 = 2.0 * a * b
        cos_hr3 = 4.0 * a * a * a - 3.0 * a
        sin_hr3 = 3.0 * b - 4.0 * b * b * b
        
        m = (11.34072 * a + 16.46899 * cos_hr2 + 7.88380 * cos_hr3 +
            14.66441 * b - 6.37224 * sin_hr2 + 9.19364 * sin_hr3 + 77.12896)
        
        return m * self.odt_params.chroma_compress_scale

    
    def reach_m_from_table(self, h: np.ndarray, table: np.ndarray) -> np.ndarray:
        """
        Vectorized version of reach_m_from_table
        
        Args:
            h: Array of hue values
            table: Reach M table
            
        Returns:
            Interpolated values from table
        """
        table_len = len(table)
        step = 360.0 / table_len
        
        # Calculate lower indices
        i_lo = np.floor(h / step).astype(int) % table_len
        
        # Calculate higher indices
        i_hi = (i_lo + 1) % table_len
        
        # Calculate interpolation factor
        t = (h - i_lo * step) / step
        
        # Get values from table
        v_lo = table[i_lo]
        v_hi = table[i_hi]
        
        # Linear interpolation
        return v_lo + t * (v_hi - v_lo)
    
    def toe(self, x: np.ndarray, limit: np.ndarray, k1_in: np.ndarray, k2_in: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Vectorized toe function for batch processing
        
        Args:
            x: Input values array
            limit: Limit values array
            k1_in: First parameter array
            k2_in: Second parameter array
            invert: Whether to use inverse calculation
            
        Returns:
            Processed values array
        """
        # Create a copy to avoid modifying the input
        result = np.copy(x)
        
        # Create mask for values <= limit
        mask = (x <= limit)
        
        # If no values need processing, return early
        if not np.any(mask):
            return result
        
        # Process only values that need toe application
        k2 = np.maximum(k2_in[mask], 0.001)
        k1 = np.sqrt(k1_in[mask] * k1_in[mask] + k2 * k2)
        k3 = (limit[mask] + k1) / (limit[mask] + k2)
        
        if invert:
            # Apply inverse toe function
            result[mask] = (x[mask] * x[mask] + k1 * x[mask]) / (k3 * (x[mask] + k2))
        else:
            # Apply forward toe function
            minus_b = k3 * x[mask] - k1
            minus_c = k2 * k3 * x[mask]
            result[mask] = 0.5 * (minus_b + np.sqrt(minus_b * minus_b + 4.0 * minus_c))
        
        return result
    
    def compress_gamut(self, jmh: np.ndarray, jx: np.ndarray, gamut_cusp_table: np.ndarray,
                   gamut_top_gamma: np.ndarray, reach_table: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Compress colors to fit within gamut boundaries.
        
        Args:
            jmh: Array of shape (-1, 3) containing JMh values
            jx: Array of shape (-1,) containing jx values
            gamut_cusp_table: Gamut cusp table
            gamut_top_gamma: Gamut top gamma values
            reach_table: Reach table
            invert: Whether to invert the compression
            
        Returns:
            Compressed JMh values as array of shape (-1, 3)
        """
        COMPRESSION_THRESHOLD = 0.75
        CUSP_MID_BLEND = 1.3
        
        # Extract components
        j = jmh[:, 0]
        m = jmh[:, 1]
        h = jmh[:, 2]
        
        # Get cusps for all hues
        jm_cusp = self.cusp_from_table(h, gamut_cusp_table)
        cusp_j = jm_cusp[:, 0]
        cusp_m = jm_cusp[:, 1]
        
        # Create result array (copy of input)
        result = np.copy(jmh)
        
        # Early return mask for zero chroma or high J
        early_return_mask = (m < 0.0001) | (j > self.odt_params.limit_jmax)
        result[early_return_mask, 1] = 0.0  # Set M to 0 for early returns
        
        # Process only non-early-return pixels
        process_mask = ~early_return_mask
        if not np.any(process_mask):
            return result
        
        # Focus J calculation
        focus_j = self.lerp(
            cusp_j[process_mask], 
            self.odt_params.mid_j, 
            np.minimum(1.0, CUSP_MID_BLEND - (cusp_j[process_mask] / self.odt_params.limit_jmax))
        )
        
        # Slope gain calculation
        slope_gain = self.odt_params.limit_jmax * self.odt_params.focus_dist * self.get_focus_gain(
            jx[process_mask], 
            cusp_j[process_mask], 
            self.odt_params.limit_jmax
        )
        
        # Get gamma values
        gamma_top = self.hue_dependent_upper_hull_gamma(h[process_mask], gamut_top_gamma)
        gamma_bottom = self.odt_params.lower_hull_gamma
        
        # Find boundary intersections
        boundary_return = self.find_gamut_boundary_intersection(
            jmh[process_mask], 
            np.column_stack((cusp_j[process_mask], cusp_m[process_mask])), 
            focus_j, 
            self.odt_params.limit_jmax, 
            slope_gain, 
            gamma_top, 
            gamma_bottom
        )
        
        j_boundary = boundary_return[:, 0]
        m_boundary = boundary_return[:, 1]
        j_intersect_source = boundary_return[:, 2]
        
        # Get reach boundary
        reach_boundary = self.get_reach_boundary(
            j_boundary, 
            m_boundary, 
            h[process_mask], 
            np.column_stack((cusp_j[process_mask], cusp_m[process_mask])), 
            focus_j, 
            reach_table
        )
        
        # Calculate compression parameters
        difference = np.maximum(1.0001, reach_boundary[:, 1] / m_boundary)
        threshold = np.maximum(COMPRESSION_THRESHOLD, 1.0 / difference)
        v = m[process_mask] / m_boundary
        
        # Apply compression function
        v = self.compression_function(v, threshold, difference, invert)
        
        # Calculate compressed values
        j_compressed = j_intersect_source + v * (j_boundary - j_intersect_source)
        m_compressed = 0.0 + v * m_boundary  # project_to[1] is 0.0
        
        # Update result array
        result[process_mask, 0] = j_compressed
        result[process_mask, 1] = m_compressed
        
        return result
    
    def get_focus_gain(self, j: np.ndarray, cusp_j: np.ndarray, limit_jmax: float) -> np.ndarray:
        """Vectorized get_focus_gain function"""
        FOCUS_GAIN_BLEND = 0.3
        FOCUS_ADJUST_GAIN = 0.55
        
        thr = self.lerp(cusp_j, limit_jmax, FOCUS_GAIN_BLEND)
        
        # Create result array with default value 1.0
        result = np.ones_like(j)
        
        # Apply calculation only where j > thr
        mask = (j > thr)
        if np.any(mask):
            gain = (limit_jmax - thr[mask]) / np.maximum(0.0001, limit_jmax - j[mask])
            result[mask] = np.power(np.log10(gain), 1.0 / FOCUS_ADJUST_GAIN) + 1.0
        
        return result
    
    def hue_dependent_upper_hull_gamma(self, h: np.ndarray, gamma_table: np.ndarray) -> np.ndarray:
        """Vectorized hue_dependent_upper_hull_gamma function"""
        i_lo = self._hue_position_in_uniform_table(h, self.GAMUT_TABLE_SIZE) + self.BASE_INDEX
        i_hi = (i_lo + 1) % self.TOTAL_TABLE_SIZE
        
        base_hue = (i_lo - self.BASE_INDEX) * 360.0 / self.GAMUT_TABLE_SIZE
        t = (h - base_hue) / (360.0 / self.GAMUT_TABLE_SIZE)
        
        return self.lerp(gamma_table[i_lo], gamma_table[i_hi], t)

    def find_gamut_boundary_intersection(self, jmh_s: np.ndarray, jm_cusp: np.ndarray, j_focus: np.ndarray,
                                    j_max: float, slope_gain: np.ndarray, gamma_top: np.ndarray, 
                                    gamma_bottom: float) -> np.ndarray:
        """Vectorized find_gamut_boundary_intersection function"""
        SMOOTH_CUSPS = 0.12
        SMOOTH_M = 0.27
        
        s = np.maximum(0.000001, SMOOTH_CUSPS)
        
        # Adjust cusp values
        cusp_j = jm_cusp[:, 0]
        cusp_m = jm_cusp[:, 1] * (1.0 + SMOOTH_M * s)
        
        # Solve for intersections
        j_intersect_source = self.solve_j_intersect(jmh_s[:, 0], jmh_s[:, 1], j_focus, j_max, slope_gain)
        j_intersect_cusp = self.solve_j_intersect(cusp_j, cusp_m, j_focus, j_max, slope_gain)
        
        # Calculate slope
        slope = np.zeros_like(j_intersect_source)
        below_focus = (j_intersect_source < j_focus)
        
        slope[below_focus] = (j_intersect_source[below_focus] * 
                            (j_intersect_source[below_focus] - j_focus[below_focus]) / 
                            (j_focus[below_focus] * slope_gain[below_focus]))
        
        slope[~below_focus] = ((j_max - j_intersect_source[~below_focus]) * 
                            (j_intersect_source[~below_focus] - j_focus[~below_focus]) / 
                            (j_focus[~below_focus] * slope_gain[~below_focus]))
        
        # Calculate boundary values
        m_boundary_lower = (j_intersect_cusp * 
                            np.power(j_intersect_source / j_intersect_cusp, 1.0 / gamma_bottom) / 
                            (cusp_j / cusp_m - slope))
        
        m_boundary_upper = (cusp_m * (j_max - j_intersect_cusp) * 
                            np.power((j_max - j_intersect_source) / 
                                    np.maximum(1e-5, j_max - j_intersect_cusp), 1.0 / gamma_top) /
                            (slope * cusp_m + j_max - cusp_j))
        
        m_boundary = cusp_m * self._smin(m_boundary_lower / cusp_m, m_boundary_upper / cusp_m, s)
        j_boundary = j_intersect_source + slope * m_boundary
        
        return np.column_stack((j_boundary, m_boundary, j_intersect_source))
    
    def _smin(self, a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
        """
        Vectorized smooth minimum function.
        
        Args:
            a: First array of values
            b: Second array of values
            s: Smoothness parameter (scalar)
            
        Returns:
            Array of smooth minimum values
        """
        # Calculate h (smoothing factor)
        h = np.maximum(s - np.abs(a - b), 0.0) / s
        
        # Calculate and return smooth minimum
        return np.minimum(a, b) - h * h * h * s * (1.0 / 6.0)

    
    def get_reach_boundary(self, j: np.ndarray, m: np.ndarray, h: np.ndarray, 
                        jm_cusp: np.ndarray, focus_j: np.ndarray, reach_table: np.ndarray) -> np.ndarray:
        """Vectorized get_reach_boundary function"""
        reach_max_m = self.reach_m_from_table(h, reach_table)
        
        cusp_j = jm_cusp[:, 0]
        slope_gain = (self.odt_params.limit_jmax * self.odt_params.focus_dist * 
                    self.get_focus_gain(j, cusp_j, self.odt_params.limit_jmax))
        
        intersect_j = self.solve_j_intersect(j, m, focus_j, self.odt_params.limit_jmax, slope_gain)
        
        # Calculate slope
        slope = np.zeros_like(intersect_j)
        below_focus = (intersect_j < focus_j)
        
        slope[below_focus] = (intersect_j[below_focus] * 
                            (intersect_j[below_focus] - focus_j[below_focus]) / 
                            (focus_j[below_focus] * slope_gain[below_focus]))
        
        slope[~below_focus] = ((self.odt_params.limit_jmax - intersect_j[~below_focus]) * 
                            (intersect_j[~below_focus] - focus_j[~below_focus]) / 
                            (focus_j[~below_focus] * slope_gain[~below_focus]))
        
        boundary = (self.odt_params.limit_jmax * 
                np.power(intersect_j / self.odt_params.limit_jmax, self.odt_params.model_gamma) * 
                reach_max_m / (self.odt_params.limit_jmax - slope * reach_max_m))
        
        return np.column_stack((j, boundary, h))
    
    def compression_function(self, v: np.ndarray, thr: np.ndarray, lim: np.ndarray, invert: bool = False) -> np.ndarray:
        """Vectorized compression_function"""
        s = (lim - thr) * (1.0 - thr) / (lim - 1.0)
        nd = (v - thr) / s
        
        # Create result array (copy of input)
        result = np.copy(v)
        
        if invert:
            # Apply only where needed
            mask = (v >= thr) & (lim > 1.0001) & (v <= thr + s)
            if np.any(mask):
                result[mask] = thr[mask] + s[mask] * (-nd[mask] / (nd[mask] - 1))
        else:
            # Apply only where needed
            mask = (v >= thr) & (lim > 1.0001)
            if np.any(mask):
                result[mask] = thr[mask] + s[mask] * nd[mask] / (1.0 + nd[mask])
        
        return result
    
    def cusp_from_table(self, h: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Vectorized cusp_from_table function"""
        i = self._hue_position_in_uniform_table(h, self.GAMUT_TABLE_SIZE) + self.BASE_INDEX
        
        # Get values from table
        lo_j = table[i, 0]
        lo_m = table[i, 1]
        lo_h = table[i, 2]
        
        hi_j = table[(i + 1) % len(table), 0]
        hi_m = table[(i + 1) % len(table), 1]
        hi_h = table[(i + 1) % len(table), 2]
        
        # Calculate interpolation factor
        t = (h - lo_h) / (hi_h - lo_h)
        
        # Interpolate values
        cusp_j = self.lerp(lo_j, hi_j, t)
        cusp_m = self.lerp(lo_m, hi_m, t)
        
        return np.column_stack((cusp_j, cusp_m))
    
    def lerp(self, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Vectorized linear interpolation"""
        return a + t * (b - a)

    def solve_j_intersect(self, j: np.ndarray, m: np.ndarray, focus_j: np.ndarray, max_j: float, slope_gain: np.ndarray) -> np.ndarray:
        """
        Vectorized version of solve_j_intersect
        
        Args:
            j: Array of j values
            m: Array of m values
            focus_j: Array of focus_j values
            max_j: Maximum j value (scalar)
            slope_gain: Array of slope_gain values
            
        Returns:
            Array of intersection j values
        """
        a = m / (focus_j * slope_gain)
        
        # Create b and c arrays with correct shape
        b = np.zeros_like(j)
        c = np.zeros_like(j)
        
        # Apply different formulas based on condition j < focus_j
        below_focus = (j < focus_j)
        
        # Calculate b values
        b[below_focus] = 1.0 - m[below_focus] / slope_gain[below_focus]
        b[~below_focus] = -(1.0 + m[~below_focus] / slope_gain[~below_focus] + 
                            max_j * m[~below_focus] / (focus_j[~below_focus] * slope_gain[~below_focus]))
        
        # Calculate c values
        c[below_focus] = -j[below_focus]
        c[~below_focus] = max_j * m[~below_focus] / slope_gain[~below_focus] + j[~below_focus]
        
        # Calculate root
        root = np.sqrt(b * b - 4.0 * a * c)
        
        # Calculate result based on condition
        result = np.zeros_like(j)
        result[below_focus] = 2.0 * c[below_focus] / (-b[below_focus] - root[below_focus])
        result[~below_focus] = 2.0 * c[~below_focus] / (-b[~below_focus] + root[~below_focus])
        
        return result
    
    def _hue_position_in_uniform_table(self, hue: np.ndarray, table_size: int) -> np.ndarray:
        """
        Vectorized version of _hue_position_in_uniform_table
        
        Args:
            hue: Array of hue values
            table_size: Size of the table
            
        Returns:
            Array of integer positions in the table
        """
        wrapped_hue = hue % 360.0
        
        # Handle negative hues
        wrapped_hue = np.where(wrapped_hue < 0, wrapped_hue + 360.0, wrapped_hue)
        
        # Calculate and return integer positions
        return np.floor(wrapped_hue / 360.0 * table_size).astype(np.int32)

    def tonescale_fwd(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized forward tonescale mapping from scene-referred to display-referred.
        
        Args:
            x: Array of input values
            
        Returns:
            Array of tonescale-mapped values
        """
        # Ensure x is non-negative for the power operation
        x_safe = np.maximum(0.0, x)
        
        # Calculate f
        f = self.ts_params.m_2 * np.power(x_safe / (x + self.ts_params.s_2), self.ts_params.g)
        
        # Calculate h
        h = np.maximum(0.0, f * f / (f + self.ts_params.t_1))
        
        # Return scaled result
        return h * self.ts_params.n_r

    def tonescale_inv(self, y: np.ndarray) -> np.ndarray:
        """
        Vectorized inverse tonescale mapping from display-referred to scene-referred.
        
        Args:
            y: Array of input values
            
        Returns:
            Array of inverse tonescale-mapped values
        """
        # Clamp input values
        z = np.maximum(0.0, np.minimum(self.ts_params.n / (self.ts_params.u_2 * self.ts_params.n_r), y))
        
        # Calculate h
        h = (z + np.sqrt(z * (4.0 * self.ts_params.t_1 + z))) / 2.0
        
        # Calculate f
        f = self.ts_params.s_2 / (np.power((self.ts_params.m_2 / h), (1.0 / self.ts_params.g)) - 1.0)
        
        return f

    def _y_to_sucs_j(self, y: np.ndarray) -> np.ndarray:
        """
        Vectorized conversion from Y to SUCS J.
        
        Args:
            y: Array of Y values
            
        Returns:
            Array of SUCS J values
        """
        return ((y / 100) ** 0.43) * 100

    def _j_to_sucs_y(self, j: np.ndarray) -> np.ndarray:
        """
        Vectorized conversion from SUCS J to Y.
        
        Args:
            j: Array of SUCS J values
            
        Returns:
            Array of Y values
        """
        return ((j / 100) ** (1 / 0.43)) * 100
    
    def ODT_fwd(self, jmh: np.ndarray) -> np.ndarray:
        """
        Vectorized forward ODT mapping from scene-referred to display-referred.
        
        Args:
            jmh: Array of shape (-1, 3) containing JMh values
            
        Returns:
            Array of shape (-1, 3) containing display-referred JMh values
        """
        # Extract components
        j = jmh[:, 0]
        m = jmh[:, 1]
        h = jmh[:, 2]
        
        # Convert J to Y and apply tonescale
        y = self._j_to_sucs_y(j) / self.odt_params.reference_luminance
        tonemapped_y = self.tonescale_fwd(y)
        tonemapped_j = self._y_to_sucs_j(tonemapped_y)
        # Create tonemapped JMh array
        tonemapped_jmh = np.column_stack((tonemapped_j, m, h))
        tonemapped_jmh = jmh
        # Apply chroma compression
        tonemapped_jmh[:, 1] = self.chroma_compression(tonemapped_jmh, j, self.reachm_table)
        
        # Apply gamut compression and return
        return self.compress_gamut(tonemapped_jmh, tonemapped_jmh[:, 0], 
                                self.gamut_cusp_table, self.gamut_top_gamma, 
                                self.reachm_table)

    def ODT_inv(self, jmh: np.ndarray) -> np.ndarray:
        """
        Vectorized inverse ODT mapping from display-referred to scene-referred.
        
        Args:
            jmh: Array of shape (-1, 3) containing display-referred JMh values
            
        Returns:
            Array of shape (-1, 3) containing scene-referred JMh values
        """
        # 1. First reverse gamut compression (opposite of the last step in ODT_fwd)
        uncompressed_gamut_jmh = self.compress_gamut(jmh, jmh[:, 0], 
                                                self.gamut_cusp_table, 
                                                self.gamut_top_gamma, 
                                                self.reachm_table, 
                                                invert=True)
        
        # Convert J to Y and apply inverse tonescale
        y = self._j_to_sucs_y(uncompressed_gamut_jmh[:, 0]) / self.odt_params.reference_luminance
        scene_y = self.tonescale_inv(y)
        scene_j = self._y_to_sucs_j(scene_y * self.odt_params.reference_luminance)
        
        # 2. Then reverse chroma compression (opposite of the second-to-last step in ODT_fwd)
        uncompressed_chroma = self.chroma_compression(uncompressed_gamut_jmh, 
                                                    scene_j, 
                                                    self.reachm_table, 
                                                    invert=True)
        
        # Create and return the final scene-referred JMh array
        return np.column_stack((scene_j, uncompressed_chroma, uncompressed_gamut_jmh[:, 2]))
