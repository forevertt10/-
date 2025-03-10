from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Untitled-1.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return "未上传文件", 400
    
    file = request.files['image']
    if file.filename == '':
        return "未选择文件", 400
    
    # 检查文件类型
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        return "文件类型不支持", 400
    
    try:
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        blurred_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if blurred_img is None:
            return "无法解码图像，请检查文件格式", 400
        
        # 增强图像
        sharpened_img = sharpen_image(blurred_img)

        # 将原图和处理后的图像保存到内存
        ext = file.filename.split('.')[-1].lower()
        if ext in ('png', 'jpg', '.jpeg', '.bmp'):
            mimetype = f'image/{ext}' if ext != 'jpg' else 'image/jpeg'
            _, buffer_original = cv2.imencode(f".{ext}", blurred_img)
            _, buffer_sharpened = cv2.imencode(f".{ext}", sharpened_img)
        else:
            return "文件类型不支持", 400
        
        # 返回JSON数据，包含两张图片的URL
        return {
            'original': f"data:{mimetype};base64,{base64.b64encode(buffer_original).decode('utf-8')}",
            'sharpened': f"data:{mimetype};base64,{base64.b64encode(buffer_sharpened).decode('utf-8')}"
        }
    except Exception as e:
        return f"处理失败: {str(e)}", 500

def sharpen_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    
    # 将梯度转换为uint8类型
    grad = cv2.convertScaleAbs(grad)
    
    # 将梯度信息加回到原图像
    sharpened = cv2.addWeighted(gray, 1.0, grad, -0.5, 0)
    
    # 将结果转换回BGR格式
    sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return sharpened

if __name__ == '__main__':
    import webbrowser
    import threading
    
    def open_browser():
        webbrowser.open('http://127.0.0.1:5000')
    
    # 启动线程
    threading.Thread(target=open_browser).start()
    
    app.run(debug=False)