from flask import Flask, jsonify, request, render_template, send_from_directory
import os

from util.postprocess import RgbToImg
from util.util import ColorOperations
from util.preprocess import readrawfile

app = Flask(__name__, static_folder='static')

# 确保目录存在
os.makedirs('static/output', exist_ok=True)
os.makedirs('raw_images', exist_ok=True)

@app.route('/')
def index():
    """渲染主页，传入原始图像列表"""
    raw_images = get_raw_image_list()
    return render_template('index.html', raw_images=raw_images)

def get_raw_image_list():
    """获取raw_images目录中的所有DNG文件"""
    images = []
    raw_dir = 'raw_images'
    
    if os.path.isdir(raw_dir):
        for file in os.listdir(raw_dir):
            if file.lower().endswith('.dng'):
                images.append(file)
    
    return images

@app.route('/static/output/<path:filename>')
def serve_image(filename):
    """提供处理后的图片文件，添加缓存控制"""
    response = send_from_directory('static/output', filename)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/process_image', methods=['POST'])
def process_image():
    """处理选中的原始图片并应用参数"""
    try:
        data = request.json
        reference_luminance = float(data.get('reference_luminance', 1600.0))
        peak_luminance = float(data.get('peak_luminance', 1538.0))
        raw_image_name = data.get('raw_image_name', '')
        
        if not raw_image_name:
            return jsonify({
                'success': False,
                'message': "未选择原始图像文件"
            })
        
        # 构建原始文件路径
        raw_file_path = os.path.join('raw_images', raw_image_name)
        
        if not os.path.exists(raw_file_path):
            return jsonify({
                'success': False,
                'message': f"原始文件 {raw_file_path} 不存在"
            })
        
        # 构建输出文件路径 (将.dng替换为.avif)
        base_name = os.path.splitext(raw_image_name)[0]
        output_file_name = f"{base_name}.avif"
        output_path = os.path.join('static', 'output', output_file_name)
        
        # 处理图像
        success = process_image_with_params(
            raw_file_path, 
            reference_luminance, 
            peak_luminance, 
            output_path
        )
        
        if success:
            return jsonify({
                'success': True,
                'output_file': output_file_name
            })
        else:
            return jsonify({
                'success': False,
                'message': "图像处理失败"
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

def process_image_with_params(raw_file_path, reference_luminance, peak_luminance, output_path):
    """使用给定参数处理图像"""
    try:
        # 设置参数
        params = {
            "output_mode": "HDR",
            "gamma": 2.2,
            "color_space": "sRGB",
            "hdr_format": "AVIF",
            "output_path": output_path
        }

        # 设置色彩空间参数
        limiting_primaries = ColorOperations.Chromaticities(
            red=[0.708, 0.292], 
            green=[0.170, 0.797], 
            blue=[0.131, 0.046], 
            white=[0.3127, 0.3290]
        )
        encoding_primaries = ColorOperations.Chromaticities(
            red=[0.64, 0.33], 
            green=[0.30, 0.60], 
            blue=[0.15, 0.06], 
            white=[0.3127, 0.3290]
        )

        # 读取原始图像
        raw_img = readrawfile(raw_file_path)
        
        # 创建颜色操作对象，使用传入的参数
        color_ops = ColorOperations(
            reference_luminance=reference_luminance, 
            peak_luminance=peak_luminance, 
            limiting_primaries=limiting_primaries, 
            encoding_primaries=encoding_primaries, 
            viewing_conditions=1
        )

        # 处理图像
        m, n = raw_img.shape[:2]
        jch = color_ops.rgb_to_jch_vectorized(raw_img.reshape(-1, 3))
        jch_compressed = color_ops.ODT_fwd(jch)
        rgb_compressed = color_ops.jch_to_rgb_vectorized(jch_compressed)
        rgb_compressed = rgb_compressed.reshape(m, n, 3) / 10000
        
        # 保存图像
        rgb_to_img = RgbToImg(
            mode=params["output_mode"], 
            gamma=params["gamma"], 
            color_space=params["color_space"], 
            hdr_format=params["hdr_format"]
        )
        rgb_to_img.process(rgb_compressed, params["output_path"])
        
        return True
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return False

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
