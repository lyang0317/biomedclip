import cv2
import torch
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import numpy as np

def extract_frame_from_video(video_path, frame_time=0):
    """
    从视频中提取指定时间的帧

    Args:
        video_path: 视频文件路径
        frame_time: 提取的时间点（秒）
    """
    # 使用 OpenCV 打开视频
    cap = cv2.VideoCapture(video_path)

    # 设置到指定时间点
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = int(frame_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # 读取帧
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError("无法从视频 " + video_path + " 中读取帧")

    # 转换 BGR 到 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 转换为 PIL Image
    pil_image = Image.fromarray(frame_rgb)

    return pil_image

def classify_video_frame(video_path, modality='ultrasound', frame_time=2):
    """
    对视频文件进行分类（通过提取帧）
    """
    # 加载模型
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # 从视频提取帧
    frame_image = extract_frame_from_video(video_path, frame_time)

    # 使用之前定义的分类函数
    result = classify_medical_image_frame(model, preprocess, tokenizer, frame_image, modality, device)

    return result

def classify_medical_image_frame(model, preprocess, tokenizer, image, modality, device):
    """
    对单帧图像进行分类
    """
    normal_prompts = [
        'this is a normal ' + modality + ' image',
        'healthy ' + modality + ' scan',
        'unremarkable ' + modality + ' finding',
        'no abnormality in this ' + modality,
        'normal findings in ' + modality
    ]

    abnormal_prompts = [
        'this is an abnormal ' + modality + ' image',
        'pathological ' + modality + ' scan',
        'abnormal finding in ' + modality,
        'diseased ' + modality + ' image',
        modality + ' showing pathology'
    ]

    # 预处理图像
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Tokenize文本
    all_texts = normal_prompts + abnormal_prompts
    texts = tokenizer(all_texts, context_length=256).to(device)

    # 模型推理
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image_tensor, texts)

        similarity = torch.matmul(image_features, text_features.t())
        logits = (logit_scale * similarity).softmax(dim=-1)

        normal_probs = logits[0, :len(normal_prompts)].mean().item()
        abnormal_probs = logits[0, len(normal_prompts):].mean().item()

        total = normal_probs + abnormal_probs
        normal_score = normal_probs / total
        abnormal_score = abnormal_probs / total

        return {
            'normal_probability': normal_score,
            'abnormal_probability': abnormal_score,
            'prediction': 'normal' if normal_score > abnormal_score else 'abnormal',
            'confidence': max(normal_score, abnormal_score)
        }

# 使用示例
if __name__ == "__main__":

    for i in range(1, 112):
        video_path = 'G:\\ML_DATA\\4ch视频二分类数据集\\4ch_binary_dataset_b_mode\\normal\\' + str(i) + '.avi'

        try:
            result = classify_video_frame(video_path, modality='ultrasound', frame_time=2)
            if result['prediction'] == 'abnormal' :
                print(str(i) + "分类结果: " + result['prediction'])
                print(str(i) + "正常概率: %.3f" % result['normal_probability'])
                print(str(i) + "异常概率: %.3f" % result['abnormal_probability'])
                print(str(i) + "置信度: %.3f" % result['confidence'])
        except Exception as e:
            print("处理视频时出错: " + str(e))