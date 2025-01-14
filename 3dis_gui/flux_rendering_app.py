import os
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS  # Import the flask_cors module to handle cross-domain requests
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from threeDIS.utils import seed_everything
import json, time
import numpy as np
from PIL import Image
import torch

last_ckpt_path_lock = threading.Lock()  # Ensure thread-safe updates of last_ckpt_path
last_ckpt_path = None
have_load_pipe = False


app = Flask(__name__)  # Initialize Flask application
CORS(app)  # Use the CORS decorator to enable cross-origin resource sharing for all routes
executor = ThreadPoolExecutor(max_workers=1)  # Make sure only one request is processed at a time

# Define the root route, used to serve static files (such as front-end pages)
@app.route('/')
def index():
    # Use the send_from_directory function to send the 'base.html' file from the 'templates' directory
    return send_from_directory('templates', 'flux_rendering.html')

GUI_progress = [100]  # Global variables are used to track progress bar

pipe = None
sam_predictor = None

control_image = None

@app.route('/GUI_progress')
def progress_updates():
    def generate():
        global GUI_progress
        while GUI_progress[0] < 100:
            json_data = json.dumps({'GUI_progress': GUI_progress[0]})
            yield f"data:{json_data}\n\n"
            time.sleep(0.1)
        yield "data:{\"GUI_progress\": 100}\n\n"
    GUI_progress[0] = 0
    return Response(generate(), mimetype='text/event-stream')


def process_request(req_data):
    data = req_data['prompt']
    
    InstanceNum = data['InstanceNum']
    use_sam_enhance = data["UseSAM"]
    width = data['width']
    height = data['height']
    num_inference_steps = int(data["num_inference_steps"])
    prompt_final = [[data['positive_prompt']]]
    negative_prompt = "worst quality, low quality, bad anatomy, " + data['negative_prompt']
    bboxes = [[]]
    for i in range(1, InstanceNum + 1):
        InstanceData = data[f'Instance{i}']['inputs']
        prompt_final[0].append(InstanceData['text'])
        prompt_final[0][0]  += ',' + InstanceData['text']
        l = InstanceData['x'] / width
        u = InstanceData['y'] / height
        r = l + InstanceData['width'] / width
        d = u + InstanceData['height'] / height
        bboxes[0].append([l, u, r, d])
    gamma = int(data['gamma'])
    global pipe
    global sam_predictor
    
    app_file_path = __file__
    app_folder = os.path.dirname(app_file_path)
    output_folder = os.path.join(app_folder, 'output_images')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    global control_image
    from PIL import Image
    if control_image is None:
        image = Image.open(os.path.join(output_folder, "depth_warning.png"))
        image.save(os.path.join(output_folder, "out.png"))
        return "out"


    with last_ckpt_path_lock:
        global have_load_pipe
        if not have_load_pipe:
            from threeDIS.utils import ConstructFluxRenderingPipeline, ConstructSAM2Predictor
            pipe = ConstructFluxRenderingPipeline()
            sam_predictor = ConstructSAM2Predictor()
            
            have_load_pipe = True
    

    seed = int(data['seed'])
    cfg = float(data['cfg'])
    seed_everything(seed)
    print(prompt_final)
    print(bboxes)
    from PIL import Image
    import base64
    import io

    

    print(data)

    print(data.keys())
    # print(data['control_image'])
    print('Generating Image..')
    prompt = prompt_final[0][0]
    prompt_2 = '$BREAKFLAG$'.join(prompt_final[0])
    instance_box_list = bboxes[0]



    image = pipe(
        prompt=prompt,
        prompt_2=prompt_2,
        control_image=control_image,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=cfg,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = instance_box_list,
        hard_control_steps = num_inference_steps,
        I2I_control_steps = gamma,
        I2T_control_steps = num_inference_steps,
        T2T_control_steps = num_inference_steps,
        T2I_control_steps = num_inference_steps,
        use_sam_enhance = use_sam_enhance,
        sam_predictor = sam_predictor if use_sam_enhance else None,
        GUI_progress=GUI_progress
        # instance_token_num = args.instance_token_num
    ).images[0]

    
    image.save(os.path.join(output_folder, "out.png"))
    return "out"


@app.route('/get_image', methods=['POST'])
def get_image():
    data = request.json
    # Add request data to queue
    future = executor.submit(process_request, data)
    fig_name = future.result()  # Block and wait until image processing is completed
    return send_from_directory('output_images', f'{fig_name}.png')

@app.route('/set_depth_map', methods=['POST'])
def set_depth_map():
    data = request.json
    data = data['prompt']
    print(data.keys())
    data_url = data["c_image"]


    import base64
    import io
    # 解码Base64图像数据（去除"data:image/png;base64,"部分）
    # 通常这个字符串会以"data:image/png;base64,"开始，所以需要从逗号后开始解码
    if "base64," in data_url:
        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
    else:
        # 错误处理：数据格式不正确
        print("Error: Data URL does not seem to be valid Base64.")

    # 使用解码的图像数据创建一个Pillow图像对象
    global control_image
    control_image = Image.open(io.BytesIO(image_data))
    control_image = control_image.convert('RGB')
    control_image.save('./debug.png')
    return 'hehe'


@app.route('/get_sd_ckpts', methods=['POST'])
def get_sd_ckpts():
    directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'migc_gui_weights/sd')
    files = [f for f in os.listdir(directory) if f.endswith('.safetensors')]
    print(files)
    return jsonify(files)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3344)
    args = parser.parse_args()
    # Start the Flask application and enable debugging mode
    app.run(debug=True, port=args.port)