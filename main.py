import argparse
from util.postprocess import RgbToImg
from util.util import ColorOperations
from util.preprocess import readrawfile

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert RAW images to SDR/HDR formats.")
    parser.add_argument("input", type=str, help="Path to the input RAW file")
    parser.add_argument("--format", type=str, choices=["jpeg", "png", "avif", "heic"], default="avif", help="Output image format")
    parser.add_argument("--eotf", type=str, choices=["gamma", "pq", "hlg", "gainmap"], default="pq", help="Electro-Optical Transfer Function (EOTF)")
    parser.add_argument("--reference_luminance", type=float, default=100, help="Display brightness (nits)")
    parser.add_argument("--max_luminance", type=float, default=1000, help="Scene brightness (nits)")
    parser.add_argument("--output", type=str, default="output", help="Output file name (without extension)")
    args = parser.parse_args()

    # Set parameters based on command-line arguments
    params = {
        "raw_file_path": args.input,
        "color_space": "sRGB",
        "output_mode": "HDR" if args.eotf != "gamma" else "SDR",
        "image_name": args.output,
        "hdr_format": args.format.upper() if args.format != "jpeg" else "JPEG",
        "output_path": f"{args.output}.{args.format}",
        "gamma": 2.2 if args.eotf == "gamma" else None,
        "demosaic": False,
    }

    # Define chromaticities
    limiting_primaries = ColorOperations.Chromaticities(red=[0.708, 0.292], green=[0.170, 0.797], blue=[0.131, 0.046], white=[0.3127, 0.3290])
    encoding_primaries = ColorOperations.Chromaticities(red=[0.64, 0.33], green=[0.30, 0.60], blue=[0.15, 0.06], white=[0.3127, 0.3290])

    # Read the RAW file
    raw_img = readrawfile(args.input)

    # Initialize ColorOperations
    color_ops = ColorOperations(
        reference_luminance=args.reference_luminance,
        peak_luminance=args.max_luminance,
        limiting_primaries=limiting_primaries,
        encoding_primaries=encoding_primaries,
        viewing_conditions=1
    )

    # Process the image
    m, n = raw_img.shape[:2]
    jch = color_ops.rgb_to_jch_vectorized(raw_img.reshape(-1, 3))
    jch_compressed = color_ops.ODT_fwd(jch)
    rgb_compressed = color_ops.jch_to_rgb_vectorized(jch_compressed)
    rgb_compressed = rgb_compressed.reshape(m, n, 3) / 10000

    # Convert to the desired output format
    rgb_to_img = RgbToImg(
        mode=params["output_mode"],
        gamma=params["gamma"],
        color_space=params["color_space"],
        hdr_format=params["hdr_format"]
    )
    rgb_to_img.process(rgb_compressed, params["output_path"])

    print(f"Image saved as {params['output_path']}")

if __name__ == "__main__":
    main()
