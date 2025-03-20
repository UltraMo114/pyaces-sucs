
# Py-ACES-SUCS

A Python implementation of **aces-core**, providing a streamlined workflow to convert RAW images into display-ready SDR/HDR images with consistent cross-device viewing experiences.

![Demo](https://via.placeholder.com/800x400.png?text=HDR/SDR+Comparison+Demo)
*(Replace with actual demo image)*

## Key Features
- **RAW to SDR/HDR Conversion**: Convert RAW images to display-optimized formats:
  - **SDR**: JPEG/PNG with Gamma OETF
  - **HDR**: HEIC/AVIF with PQ, HLG, or GainMap EOTF
- **ACES Workflow**: Full implementation of ACES color management:
  - **IDT (Input Device Transform)** for camera-specific RAW processing
  - **sUCS-based ODT (Output Device Transform)** for perceptual uniformity
- **Web UI**: Flask-based interface for real-time preview (supports HDR in Chrome)
- **Display-Agnostic Output**: Uses ACES reference luminance model to maximize display capabilities:
  - `reference luminance`: Screen's native brightness
  - `max luminance`: Scene's original brightness

## Why ACES?
ACES eliminates the SDR/HDR dichotomy by:
1. Separating scene-referred (**max luminance**) and display-referred (**reference luminance**) brightness
2. Preserving scene intent across displays without artificial contrast boosting
3. Supporting true HDR when `max luminance > reference luminance`

## Why sUCS?
Our ODT implementation uses **sUCS/sCAM** instead of Hellwig-JMh/CAM16 because:
- Fixes CAM16's flawed RGB-based chromatic adaptation (sUCS uses LMS cone response)
- Simplified computation with two-stage adaptation:
  1. Cone intensity adaptation
  2. Chroma intensity adaptation
- Avoids impractical La/Yb estimation (fixed at 100/20 nits in traditional ACES)

## Installation
```bash
pip install opencv-python flask rawpy numpy pillow_heif Pillow
```

## Usage
### 1. Command Line Tool (main.py)
```bash
python main.py input.raw --format avif --eotf pq --max_luminance 1000
```
**Supported Options**:
- `--format`: jpeg/png/avif/heic
- `--eotf`: gamma/pq/hlg/gainmap (SDR defaults to gamma)
- `--reference_luminance`: Display brightness (nits), default=100
- `--max_luminance`: Scene brightness (nits), default=4000

### 2. Web Interface (app.py)
```bash
python app.py
```
1. Place RAW files in `raw_images/`
2. Open `http://127.0.0.1:5000` in **Chrome** (HDR support required)
3. Select output parameters and preview

## Technical Implementation
1. **RAW Processing** (via rawpy):
   - Demosaicing
   - IDT application
2. **ACES Central Space Conversion**:
   - Scene-referred linear RGB
3. **sUCS ODT**:
   ```python
   def sucs_odt(image, reference_lum, max_lum):
       # 1. LMS cone adaptation
       # 2. Chroma compression
       # 3. EOTF encoding
       return display_encoded_image
   ```
4. **Format Encoding**:
   - PQ/HLG via OpenCV
   - GainMap via Pillow-HEIF

## Supported Cameras
Tested with:
- Sony α7 IV (.arw)
- Canon EOS R5 (.cr3)
- Nikon Z9 (.nef)

*(Add your camera model via issues)*

## Roadmap
- [ ] GPU acceleration
- [ ] DCTL support
- [ ] 3D LUT export

## Contributing
PRs welcome! Key development areas:
- Additional IDT profiles
- CI/CD pipeline
- Documentation

## License
MIT License (see LICENSE)
``` 

This README:
1. Uses technical terms from ACES/sUCS documentation
2. Highlights key differentiators from traditional ACES
3. Provides clear installation/usage instructions
4. Maintains scientific accuracy while being approachable
5. Includes placeholder sections (demo image, cameras) for easy customization

Would you like me to expand any specific section?
