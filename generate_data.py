from data_generator import DataGenerator
from enum import Enum
import numpy as np
import torch
import multiprocessing
import os
import json
import shutil
from datetime import datetime

class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

def run_generator(gpu_id, total_gpus, coco_json_path, input_img_dir, output_dir, all_prompts, batch_size):
    """각 GPU에서 실행될 함수"""
    # GPU 설정
    torch.cuda.set_device(gpu_id)
    print(f"Process running on GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    
    # GPU별로 다른 프롬프트 선택
    np.random.seed(gpu_id)  # GPU별로 다른 시드 설정
    prompts = np.random.choice(all_prompts, size=batch_size, replace=False)
    print(f"GPU {gpu_id} using prompts: {prompts}")
    
    # 배경 소스 랜덤 선택
    bg_source = np.random.choice([source.value for source in [BGSource.LEFT, BGSource.RIGHT, BGSource.TOP, BGSource.BOTTOM]])
    
    # GPU별 출력 디렉토리 생성
    gpu_output_dir = os.path.join(output_dir, f"gpu_{gpu_id}")
    
    # GPU별 기본 시드 설정
    base_seed = 12345 + (gpu_id * 10000)
    
    # 데이터 생성기 초기화 및 실행
    generator = DataGenerator(
        coco_json_path=coco_json_path,
        input_img_dir=input_img_dir,
        output_dir=gpu_output_dir,
        prompts=prompts,  # GPU별로 선택된 프롬프트 사용
        image_width=512,
        image_height=768,
        batch_size=batch_size,
        bg_source=bg_source,
        gpu_id=gpu_id,
        total_gpus=total_gpus,
        device=f'cuda:{gpu_id}',
        seed=base_seed
    )
    
    # 데이터 생성 실행
    generator.generate()

def merge_results(output_dir, num_gpus):
    """각 GPU의 결과를 하나로 병합"""
    print("Merging results from all GPUs...")
    
    # 최종 결과 디렉토리 생성
    final_images_dir = os.path.join(output_dir, "merged_images")
    final_anno_dir = os.path.join(output_dir, "merged_annotations")
    os.makedirs(final_images_dir, exist_ok=True)
    os.makedirs(final_anno_dir, exist_ok=True)
    
    all_images = []
    all_annotations = []
    image_id_offset = 0
    
    # 각 GPU의 결과 처리
    for gpu_id in range(num_gpus):
        gpu_dir = os.path.join(output_dir, f"gpu_{gpu_id}")
        if not os.path.exists(gpu_dir):
            print(f"Warning: Directory not found for GPU {gpu_id}: {gpu_dir}")
            continue
            
        gpu_images_dir = os.path.join(gpu_dir, "images")
        gpu_anno_dir = os.path.join(gpu_dir, "output_result_anno")
        
        # JSON 파일 찾기
        json_files = [f for f in os.listdir(gpu_dir) if f.startswith("synthetic_data_") and f.endswith(".json")]
        if not json_files:
            print(f"Warning: No JSON files found in GPU {gpu_id} directory: {gpu_dir}")
            continue
            
        json_path = os.path.join(gpu_dir, json_files[-1])
        print(f"Processing results from GPU {gpu_id}: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                gpu_data = json.load(f)
            
            if not gpu_data['images']:
                print(f"Warning: No images found in JSON file for GPU {gpu_id}")
                continue
                
            # 이미지와 어노테이션 ID 조정
            for img in gpu_data['images']:
                old_id = img['id']
                new_id = old_id + image_id_offset
                img['id'] = new_id
                
                # 이미지 파일 복사
                old_path = os.path.join(gpu_images_dir, img['file_name'])
                if not os.path.exists(old_path):
                    print(f"Warning: Image file not found: {old_path}")
                    continue
                    
                new_name = f"{new_id:06d}.png"
                new_path = os.path.join(final_images_dir, new_name)
                shutil.copy2(old_path, new_path)
                img['file_name'] = new_name
                
                # 어노테이션 이미지 복사
                old_anno_path = os.path.join(gpu_anno_dir, f"anno_{img['file_name']}")
                new_anno_path = os.path.join(final_anno_dir, f"anno_{new_name}")
                if os.path.exists(old_anno_path):
                    shutil.copy2(old_anno_path, new_anno_path)
                
                all_images.append(img)
            
            for ann in gpu_data['annotations']:
                old_img_id = int(ann['image_id'])
                ann['image_id'] = old_img_id + image_id_offset
                all_annotations.append(ann)
            
            if all_images:  # 이미지가 있을 때만 offset 업데이트
                image_id_offset = max(img['id'] for img in all_images) + 1
                
        except Exception as e:
            print(f"Error processing GPU {gpu_id} results: {e}")
            continue
    
    if not all_images:
        print("Error: No images were found to merge!")
        return
    
    # 최종 JSON 생성
    final_data = {
        'info': {
            'description': 'Merged synthetic dataset generated with IC-Light',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'licenses': [],
        'categories': gpu_data['categories'],
        'images': all_images,
        'annotations': all_annotations
    }
    
    # 최종 JSON 저장
    final_json_path = os.path.join(output_dir, f"merged_synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(final_json_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Results merged: {len(all_images)} images and {len(all_annotations)} annotations")
    
    # # 임시 GPU 디렉토리 정리 (선택사항)
    # for gpu_id in range(num_gpus):
    #     gpu_dir = os.path.join(output_dir, f"gpu_{gpu_id}")
    #     if os.path.exists(gpu_dir):
    #         shutil.rmtree(gpu_dir)

if __name__ == "__main__":
    # 설정
    coco_json_path = "dataset/annotation.json"
    input_img_dir = "dataset/images"
    output_dir = "output"
    
    # 가능한 모든 프롬프트 목록
    all_prompts = [
        "a item, in road, city",
        "a item, on sidewalk, urban street",
        "a item, near crosswalk, commercial district",
        "a item, in parking lot, shopping mall",
        "a item, at bus stop, downtown",
        "a item, in subway station, underground",
        "a item, near traffic light, intersection",
        "a item, at street corner, business district"
    ]
    
    batch_size = 7
    
    # 사용 가능한 GPU 수 확인
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # 멀티프로세스 실행
    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(
            target=run_generator,
            args=(gpu_id, num_gpus, coco_json_path, input_img_dir, output_dir, all_prompts, batch_size)
        )
        p.start()
        processes.append(p)
    
    # 모든 프로세스 완료 대기
    for p in processes:
        p.join()
    
    # 결과 병합
    merge_results(output_dir, num_gpus)