from util.postprocess import RgbToImg
from util.util import ColorOperations
from util.preprocess import readrawfile


def main():
    params = {
        "raw_file_path": "MCCC.dng",
        "rgb_to_xyz_method": "polynomial", ## greyworld = AWB + CCM, polynomial = polynomial transform
        "xyz_to_rgb_method": "default",
        "color_space": "sRGB",
        "output_mode": "HDR",
        "image_name": "output_img",
        "hdr_format": "AVIF",
        "output_path": "DSC0470.avif",
        "gamma": 2.2,
        "demosaic": False,
    }


    limiting_primaries = ColorOperations.Chromaticities(red=[0.708, 0.292], green=[0.170, 0.797], blue=[0.131, 0.046], white=[0.3127, 0.3290])
    encoding_primaries = ColorOperations.Chromaticities(red=[0.64, 0.33], green=[0.30, 0.60], blue=[0.15, 0.06], white=[0.3127, 0.3290])

    image_path = "5_2.3.dng"
    raw_img = readrawfile(image_path)
    color_ops = ColorOperations(reference_luminance=1600.0, peak_luminance=1600.0, limiting_primaries=limiting_primaries, encoding_primaries=encoding_primaries, viewing_conditions=1)

    m, n = raw_img.shape[:2]
    jch = color_ops.rgb_to_jch_vectorized(raw_img.reshape(-1, 3))
    jch_compressed = color_ops.ODT_fwd(jch)
    rgb_compressed = color_ops.jch_to_rgb_vectorized(jch_compressed)
    rgb_compressed = rgb_compressed.reshape(m, n, 3) / 10000
    rgb_to_img = RgbToImg(mode=params["output_mode"], gamma=params["gamma"], color_space=params["color_space"], hdr_format= params["hdr_format"])
    rgb_to_img.process(rgb_compressed, params["output_path"])


if __name__ == "__main__":
    main()