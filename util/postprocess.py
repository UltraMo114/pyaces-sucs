import numpy as np
from PIL import Image
import pillow_heif

# Set global options for pillow_heif
pillow_heif.options.QUALITY = -1

class RgbToImg:
    def __init__(self, mode="SDR", color_space="sRGB", gamma=2.4, hdr_format="HEIF"):
        """
        Initialize the RgbToImg module.

        Parameters:
        - mode (str): Output mode ("SDR" or "HDR").
        - color_space (str): Color space for HDR output ("sRGB", "Display P3", "BT-2020").
        - gamma (float): Gamma value for SDR output (default is 2.4 for sRGB).
        """
        self.mode = mode
        self.color_space = color_space
        self.gamma = gamma  # Gamma value for SDR output
        self.hdr_format = hdr_format  # HDR format

        # Define color primaries and transfer characteristics for HDR
        self.color_primaries_map = {
            "sRGB": 1,          # BT.709
            "Display P3": 12,   # P3-D65
            "BT-2020": 9        # BT.2020
        }

        self.transfer_characteristics_map = {
            "sRGB": 16,          # BT.709
            "Display P3": 16,   # PQ
            "BT-2020": 16       # PQ
        }

    def process(self, rgb_data, output_path):
        """
        Convert RGB data to an image file and save it.

        Parameters:
        - rgb_data (np.ndarray): RGB image (H x W x 3), normalized to [0, 1].
        - output_path (str): Path where the output image will be saved.
        """
        if self.mode == "SDR":
            self._save_sdr_image(rgb_data, output_path)
        elif self.mode == "HDR":
            self._save_hdr_image(rgb_data, output_path)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _save_sdr_image(self, rgb_data, output_path):
        """
        Save RGB data as an 8-bit JPEG image using GOG encoding with custom gamma.

        Parameters:
        - rgb_data (np.ndarray): RGB image (H x W x 3), normalized to [0, 1].
        - output_path (str): Path where the output image will be saved.
        """
        # Apply GOG encoding (Inverse Gamma Correction)
        rgb_data = np.clip(rgb_data, 0, 1)
        rgb_data = np.power(rgb_data, 1 / self.gamma)  # Inverse gamma correction
        rgb_data = np.clip(rgb_data, 0, 1)  # Ensure values are within [0, 1]

        # Scale the data to 8-bit range (0-255)
        rgb_data = (rgb_data * 255).astype(np.uint8)

        # Convert numpy array to PIL Image with 8-bit mode
        img = Image.fromarray(rgb_data, mode="RGB")  # Use "RGB" for 8-bit images

        # Save as 8-bit JPEG
        img.save(output_path, format="JPEG")

    def _save_hdr_image(self, rgb_data, output_path):
        """
        Save RGB data as an HDR HEIC/HEIF image with specified color space.

        Parameters:
        - rgb_data (np.ndarray): RGB image (H x W x 3), normalized to [0, 1].
        - output_path (str): Path where the output image will be saved.
        """
        # Get color primaries and transfer characteristics
        color_primaries = self.color_primaries_map.get(self.color_space, 1)
        transfer_characteristics = self.transfer_characteristics_map.get(self.color_space, 16)

        # Normalize the numpy array to the range [0, 1] and then scale it to [0, 65535]
        rgb_data = np.clip(rgb_data, 0, 1)
        def pq_oetf(hdr_values):
            """
            Apply PQ OETF to convert non-linear HDR values to linear luminance.

            Parameters:
            - hdr_values (np.ndarray): Non-linear HDR values (H x W x 3), normalized to [0, 1].

            Returns:
            - linear_luminance (np.ndarray): Linear luminance values (H x W x 3), normalized to [0, 1].
            """
            # Constants for PQ OETF
            c1 = 0.8359375
            c2 = 18.8515625
            c3 = 18.6875
            m2 = 78.84375
            m1 = 0.1593017578125

            # Ensure input is within valid range
            hdr_values = np.clip(hdr_values, 0, 1)

            # Apply PQ OETF
            V_m1 = np.power(hdr_values, m1)
            linear_luminance = np.power((c1 + c2 * V_m1) / (1 + c3 * V_m1), m2)

            return linear_luminance
        rgb_pq = pq_oetf(rgb_data)
        rgb_data = (rgb_pq * 65535).astype(np.uint16)
        # Create a HEIF image from the numpy array
        img = pillow_heif.from_bytes(
            mode="RGB;16",
            size=(rgb_data.shape[1], rgb_data.shape[0]),
            data=rgb_data.tobytes()
        )

        # Define the save parameters
        kwargs = {
            'format': self.hdr_format,
            'color_primaries': color_primaries,
            'transfer_characteristics': transfer_characteristics,
        }

        # Save the image to the specified output path
        img.save(output_path, **kwargs)

