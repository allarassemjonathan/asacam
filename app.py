#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, send_file
import cv2
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import argparse
import os
import time
from loguru import logger
import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

app = Flask(__name__)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
global_queue_content = []
global_queue_frames = []

def arti_parser(video_path):
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default='yolo_l')
    parser.add_argument("-n", "--name", type=str, default='yolox-l', help="model name")

    parser.add_argument(
        "--path", default=f"{video_path}", help="path to video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--app", 
        default="run"
    )
    parser.add_argument(
        "--host=0.0.0.0",
        default=""
    )
    return parser


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
    # model_dir = "C:\\Users\\ALLARASSEMJJ20\\productx\\tools\\vit-gpt2-image-captioning"
    # # Directory containing the pytorch_model.bin and config files

    # # Load the model using the directory where the .bin file is stored
    # model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # # Load the feature extractor and tokenizer
    # feature_extractor = ViTImageProcessor.from_pretrained(model_dir)
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


import json
import requests
import base64
import logging
from fpdf import FPDF
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%H:%M:%S')

# OpenAI API key = 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
api_key = "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"

# Function to encode the image
def encode_image(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return None, {"error": "Failed to encode image"}
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8'), None


# Configure logger
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%H:%M:%S')

# OpenAI API key = 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
api_key = "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"

# Function to encode the image
def encode_image(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return None, {"error": "Failed to encode image"}
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8'), None

prompt = ""
# Function to send the frame to OpenAI API
def interpret_frame_with_openai(frame):
    global prompt
    base64_image, error = encode_image(frame)
    if error:
        return error
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}. Be concise." 
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to interpret image: {response.text}")
        return {"error": response.text, "status_code": response.status_code}

contacted = False
def image_flow_demo_openai_UI_integrated(predictor, vis_folder, current_time, args):
    global contacted
    global global_queue_content
    global ind
    print('path', args.path)
    print('demo',  args.demo)
    # if the path has changed aka args.path is not video_source
    path = args.path
    cap = cv2.VideoCapture(path if args.demo == "run" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    INTERPRET_FRAME_INTERVAL = 10 # interpret every 5th frame
    collected_logs = []
    while True:
        print('LLM receives frame')
        ret_val, frame = cap.read()
        if not ret_val:
            print('restarting LLM')
            cap = cv2.VideoCapture(path if args.demo == "run" else args.camid)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)

        if ret_val:
            print('LLM processes frame')

            frame_count += 1
            #collected_content = []
            if frame_count % INTERPRET_FRAME_INTERVAL == 0:
                print('LLM sends frame to UI')
                content = interpret_frame_with_openai(frame)['choices'][0]['message']['content']
                if 'ALERT' in content:
                    # email
                    print('about to send')
                    email_send_dest(your_email, recipient_email, content)
                    contacted = True

                current_time = datetime.now().strftime('%H:%M:%S')
                global_queue_content.append(f"{current_time} :: Camera {lst_sources[args.path]} ::\n {content}")
                logger.info(f"{content}")
                #log_message = f"{current_time}: {content}"
                log_message = (current_time, content)
                collected_logs.append(log_message)
                #collected_content.append(content)
                #logger.info(f"Frame {frame_count} content: {content}")

                print('values', args.path, video_source)
                if args.path != video_source:
                    break
                print('sources', lst_sources, args.path, video_source)
                yield json.dumps({"text": f"{current_time} :: Camera {lst_sources[video_source]} ::\n {content}"})


def imageflow_demo_yolo_UI_integrated(predictor, vis_folder, current_time, args):

    print('path', args.path, 'demo', args.demo)
    cap = cv2.VideoCapture(args.path if args.demo == "run" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print('step1 ok')
    
    if not os.path.exists(args.path):
        print('path does not exist')
    else:
        print('victory')

    if not cap.isOpened():
        print('path exist but cannot be opened')
    else:
        print('victoryy')

    print('about to start')
    while True:
        print('CV processing frame')
        ret_val, frame = cap.read()

        if not ret_val:
            print('restarting CV')
            print('path', args.path, 'demo', args.demo)
            cap = cv2.VideoCapture(args.path if args.demo == "run" else args.camid)
            
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            
        if ret_val:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
    
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Initialize the camera
camera = None

video_source = None
lst_sources = {}
ind=0
current_time = None
@app.route('/video_feed')
def video_feed():
    global video_source
    global current_time
    global ind

    video_source = request.args.get('url')  # Get the URL from the query parameters
    predictor = None
    # if not video_source:
    #     video_source = "tools\\static\\videoplayback.mp4"
    # read the path from here and pass it into the args
    args = arti_parser(video_source).parse_args()
    exp = get_exp(args.exp_file, args.name)
    print('test', args.path, exp.output_dir, args.experiment_name)
    
    # keep a list of all the new cameras added to the system
    if args.path not in lst_sources:
        lst_sources[args.path]=ind
        ind+=1

    print("passing with", args.path)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    print('saving or not', args.save_result)
    if args.save_result:
        print('saving...')
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    print('starting CV ..... ')

    print(args.path)

    if 'rtsp' not in args.path: 
        while not os.path.exists(args.path):
            redirect(url_for('video_feed'))

    return Response(imageflow_demo_yolo_UI_integrated(predictor, vis_folder, current_time, args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# the solo purpose of this route is to reinitialze
# the global variable first to True

@app.route('/load_stream', methods=['POST'])
def load_stream():
    global first
    global contacted
    # helpful to restart this so 
    # that I can reload the stream
    contacted = True
    first = True
    print('restarting the LLM cycle')
    # You could include additional validation for the URL here if needed
    return jsonify(success=True)

# Set up the SMTP server
smtp_server = "smtp.gmail.com"
smtp_port = 587
your_email = "jonathanjerabe@gmail.com"
your_password = "ajrn mros lkzm urnu"
recipient_email = ""

@app.route('/prompt', methods=['POST'])
def get_prompt():
    global prompt
    global recipient_email
    data = request.get_json()
    recipient_email = data.get('email')
    prompt = data.get('prompt')
    print(f"prompt received {prompt}")
    if len(prompt)>0:
        return jsonify(success=True)
    return jsonify(success=False)

def email_send_dest(sender, dest, content):

    # Compose the email
    subject = "ASACAM REPORT"
    body = content
    print('hereemail')
    # Create the MIME message
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = dest
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Establish connection to Gmail's SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection

        # Log in to the server
        server.login(your_email, your_password)

        # Send the email
        text = msg.as_string()
        server.sendmail(your_email, recipient_email, text)

        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")

    finally:
        # Close the connection to the server
        server.quit()


@app.route('/email', methods=['POST'])
def email_reporter():
    data = request.get_json()

    # Extract the stream URL, content, and email from the received data
    content = data.get('content')
    email = data.get('email')
    reporter = data.get('reporter')
    title = data.get('title')

    # Compose the email
    recipient_email = email
    subject = "ASACAM REPORT"
    body = f'Hey {reporter},\n\n\nAn alarm has been triggered among the cameras that you own for the mission {title}. Below is the current report:'+ content + '\n\n\n' + 'we recommend you check the application as soon as possible.\n\nASACAM AUTOMATED SERVICE TEST\n\n'

    # Create the MIME message
    msg = MIMEMultipart()
    msg['From'] = your_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    # Connect to the SMTP server and send the email
    try:
        # Establish connection to Gmail's SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection

        # Log in to the server
        server.login(your_email, your_password)

        # Send the email
        text = msg.as_string()
        server.sendmail(your_email, recipient_email, text)

        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")

    finally:
        # Close the connection to the server
        server.quit()
    # You could include additional validation for the URL here if needed
    return jsonify(success=True)


# Route to get random text dynamically // THIS IS GONNA BE THE LLM TEXT
first = True
predictor_LLM = None
vis_folder_LLM = None
args_LLM = None
old_source = None
@app.route('/random_text')
def random_text():

    # these values are global because i not only need to initialize
    # them outside the loop, i also need them to conserve the state 
    # of the function.
    global old_source
    global video_source
    global first
    global predictor_LLM
    global vis_folder_LLM
    global args_LLM
    global global_queue_content
    print('1xxxx')
    print(video_source)
    if video_source==None:
        redirect(url_for('random_text'))
    print('first', first)
    if first:
        print('ok check?')
        old_source = video_source
        # if not video_source:
        #     video_source = "tools\\static\\videoplayback.mp4"
        # read the path from here and pass it into the args
        print('2xxxx')
        args_LLM = arti_parser(video_source).parse_args()
        exp = get_exp(args_LLM.exp_file, args_LLM.name)
        print('xxxx', args_LLM.path)

        if 'rtsp' not in args_LLM.path:
            while not os.path.exists(args_LLM.path):
                redirect(url_for('video_feed'))

        if not args_LLM.experiment_name:
            args_LLM.experiment_name = exp.exp_name

        file_name = os.path.join(exp.output_dir, args_LLM.experiment_name)
        os.makedirs(file_name, exist_ok=True)

        if args_LLM.save_result:
            vis_folder_LLM = os.path.join(file_name, "vis_res")
            os.makedirs(vis_folder_LLM, exist_ok=True)

        if args_LLM.trt:
            args_LLM.device = "gpu"

        logger.info("Args: {}".format(args_LLM))

        if args_LLM.conf is not None:
            exp.test_conf = args_LLM.conf
        if args_LLM.nms is not None:
            exp.nmsthre = args_LLM.nms
        if args_LLM.tsize is not None:
            exp.test_size = (args_LLM.tsize, args_LLM.tsize)

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        if args_LLM.device == "gpu":
            model.cuda()
            if args_LLM.fp16:
                model.half()  # to FP16
        model.eval()
        predictor_LLM = None
        current_time_LLM = time.localtime()
        old_source = video_source
        first=False
        return Response(image_flow_demo_openai_UI_integrated(predictor_LLM, vis_folder_LLM, current_time_LLM, args_LLM))
    
    # if source has changed and it is not the first time
    # 
    if old_source!=video_source:
        first = True
        print('second chance')
        return random_text()
  
    if global_queue_content:
        return Response(json.dumps({"text":global_queue_content[-1]}))
    return Response(json.dumps({"text":"Loading Image Analysis ... "}))
import glob
# get the last created directory
def get_last_created_directory(path):
    # Get all subdirectories in the specified path
    subdirs = [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)]

    # If there are no directories, return None
    if not subdirs:
        return None

    # Sort subdirectories by creation time (newest first)
    latest_subdir = max(subdirs, key=os.path.getctime)
    
    print(latest_subdir)
    return latest_subdir

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global args_LLM
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['pdf']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # check if the file exist
        # put the right there
        # Save the file
        
        exp = get_exp(args_LLM.exp_file, args_LLM.name)
        if exp==None:
            return jsonify({'message': 'Error', 'file_path':'None'}), 500
        if not args_LLM.experiment_name:
            args_LLM.experiment_name = exp.exp_name
        file_name = os.path.join(exp.output_dir, args_LLM.experiment_name)
        vis_folder = os.path.join(file_name, "vis_res")
        current_time = time.localtime()
        print(vis_folder)
        print('last path guy', get_last_created_directory(vis_folder))
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M", current_time))
        
        # if the image folder is already created use that instead of creating a new folder
        image_folder = get_last_created_directory(vis_folder)
        if save_folder != image_folder and image_folder!=None:
            save_folder = image_folder
            print('used the older folder')

        if os.path.exists(save_folder):
            print('I think this is it', save_folder)
        else:
            print('Does not exist oops...')
            os.makedirs(save_folder, exist_ok=True)

        save_folder = os.path.join(save_folder, 'report.pdf')
        print('final folder', save_folder)
        file.save(save_folder)
        return jsonify({'message': 'PDF saved successfully!', 'file_path':save_folder}), 200

@app.route('/vis_res/<path:folder>/<path:filename>')
def serve_file(folder, filename):
    print('test', folder, filename)
    path = os.path.join(VIS_RES_FOLDER, folder)
    path = os.path.join(path, filename)
    print(path)
    return send_file(path, as_attachment=False, conditional=False)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mission')
def index():
    return render_template('mission.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/activity')
def reports():
    return render_template('activity.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")