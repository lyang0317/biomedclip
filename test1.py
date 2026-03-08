import cv2
import torch
from PIL import Image
import numpy as np
import os
from open_clip import create_model_from_pretrained, get_tokenizer
import time

def extract_frame_from_video(video_path, frame_time=0):
    """
    从视频中提取指定时间的帧
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
        #"homogeneous echotexture",
        #"well-defined margins",
        #"normal size and shape",
        #"typical echogenicity",
        #"clear anatomical boundaries"
    ]

    abnormal_prompts = [
        'this is an abnormal ' + modality + ' image',
        'pathological ' + modality + ' scan',
        'abnormal finding in ' + modality,
        'diseased ' + modality + ' image',
        modality + ' showing pathology'
        #"heterogeneous echotexture",
        #"irregular margins",
        #"mass effect",
        #"abnormal echogenicity",
        #"loss of normal architecture"
        #"cystic lesion",
        #"solid mass",
        #"calcification with shadowing"
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

def classify_video_frame(video_path, model, preprocess, tokenizer, device, modality='ultrasound', frame_time=2):
    """
    对视频文件进行分类（通过提取帧）
    """
    # 从视频提取帧
    frame_image = extract_frame_from_video(video_path, frame_time)

    # 使用分类函数
    result = classify_medical_image_frame(model, preprocess, tokenizer, frame_image, modality, device)

    return result

def batch_classify_videos(folder_path, modality='ultrasound', frame_time=2, output_file=None):
    """
    批量处理文件夹中的所有视频文件
    """
    # 加载模型（只加载一次，提高效率）
    print("正在加载 BioMedCLIP 模型...")
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    # 加载微调后的权重
    #checkpoint = torch.load(model_path, map_location='cpu')
    #model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    print("模型加载完成，使用设备: " + str(device))

    # 支持的视频格式
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv']

    # 查找所有视频文件
    video_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = os.path.splitext(file.lower())[1]
            if file_ext in video_extensions:
                video_files.append(os.path.join(root, file))

    print("找到 %d 个视频文件" % len(video_files))

    if len(video_files) == 0:
        print("未找到视频文件，请检查文件夹路径")
        return []

    # 存储结果
    results = []

    # 批量处理
    for i, video_path in enumerate(video_files, 1):
        try:
            print("\n[%d/%d] 处理视频: %s" % (i, len(video_files), os.path.basename(video_path)))

            start_time = time.time()

            # 分类视频
            result = classify_video_frame(video_path, model, preprocess, tokenizer, device, modality, frame_time)
            result['video_path'] = video_path
            result['video_name'] = os.path.basename(video_path)
            result['processing_time'] = time.time() - start_time

            results.append(result)

            # 打印结果
            print("  分类结果: " + result['prediction'])
            print("  正常概率: %.3f" % result['normal_probability'])
            print("  异常概率: %.3f" % result['abnormal_probability'])
            print("  置信度: %.3f" % result['confidence'])
            print("  处理时间: %.2f秒" % result['processing_time'])

        except Exception as e:
            print("  处理视频 " + video_path + " 时出错: " + str(e))
            error_result = {
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'error': str(e),
                'prediction': 'error',
                'confidence': 0.0
            }
            results.append(error_result)

    # 输出统计信息
    print_summary(results)

    # 保存结果到文件
    if output_file:
        save_results_to_file(results, output_file)

    return results

def print_summary(results):
    """打印处理结果统计"""
    successful_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]

    print("\n" + "="*50)
    print("批量处理完成!")
    print("="*50)
    print("总处理视频数: %d" % len(results))
    print("成功处理: %d" % len(successful_results))
    print("处理失败: %d" % len(error_results))

    if successful_results:
        # 统计分类结果
        normal_count = sum(1 for r in successful_results if r['prediction'] == 'normal')
        abnormal_count = sum(1 for r in successful_results if r['prediction'] == 'abnormal')

        print("正常分类: %d" % normal_count)
        print("异常分类: %d" % abnormal_count)

        # 计算平均置信度
        avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
        print("平均置信度: %.3f" % avg_confidence)

        # 高置信度结果统计
        high_confidence = sum(1 for r in successful_results if r['confidence'] > 0.8)
        medium_confidence = sum(1 for r in successful_results if 0.6 <= r['confidence'] <= 0.8)
        low_confidence = sum(1 for r in successful_results if r['confidence'] < 0.6)

        print("高置信度(>0.8): %d" % high_confidence)
        print("中置信度(0.6-0.8): %d" % medium_confidence)
        print("低置信度(<0.6): %d" % low_confidence)

def save_results_to_file(results, output_file):
    """保存结果到文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("视频文件,分类结果,正常概率,异常概率,置信度,处理时间(秒),错误信息\n")

            for result in results:
                if 'error' in result:
                    f.write("%s,错误,-,-,-,-,%s\n" % (result['video_name'], result['error']))
                else:
                    f.write("%s,%s,%.3f,%.3f,%.3f,%.2f,\n" % (
                        result['video_name'],
                        result['prediction'],
                        result['normal_probability'],
                        result['abnormal_probability'],
                        result['confidence'],
                        result['processing_time']
                    ))

        print("结果已保存到: " + output_file)
    except Exception as e:
        print("保存结果文件时出错: " + str(e))

def process_separate_folders():
    """分别处理正常和异常文件夹"""
    base_path = 'G:\\ML_DATA\\4ch视频二分类数据集\\4ch_binary_dataset_b_mode'

    # 处理正常视频
    print("处理正常视频...")
    normal_results = batch_classify_videos(
        folder_path=os.path.join(base_path, 'normal'),
        modality='ultrasound',
        output_file='normal_results.csv'
    )

    # 处理异常视频
    print("\n处理异常视频...")
    abnormal_results = batch_classify_videos(
        folder_path=os.path.join(base_path, 'abnormal'),
        modality='ultrasound',
        output_file='abnormal_results.csv'
    )

    return normal_results, abnormal_results

# 使用示例
if __name__ == "__main__":
    # 设置视频文件夹路径
    video_folder = 'F:\\swpu\\ML_DATA\\4ch视频二分类数据集\\4ch_binary_dataset_b_mode\\abnormal'

    # 设置输出文件（可选）
    output_csv = 'classification_results.csv'

    try:
        # 批量处理所有视频
        results = batch_classify_videos(
            folder_path=video_folder,
            modality='ultrasound',
            frame_time=2,
            output_file=output_csv
        )

        print("\n批量推理完成！")

    except Exception as e:
        print("批量处理出错: " + str(e))