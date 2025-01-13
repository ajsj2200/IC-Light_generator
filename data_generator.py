import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import shutil
from datetime import datetime
import copy
from ic_light import get_ic_light
import cv2

class DataGenerator:
    def __init__(
        self,
        coco_json_path,
        input_img_dir,
        output_dir,
        prompts,
        image_width=512,
        image_height=768,
        batch_size=2,
        seed=12345,
        bg_source="Right Light",
        gpu_id=0,
        total_gpus=1,
        device='cuda:0'
    ):
        """
        Args:
            coco_json_path (str): COCO json 파일 경로
            input_img_dir (str): 입력 이미지 디렉토리 경로
            output_dir (str): 출력 디렉토리 경로
            prompts (list): 이미지 생성에 사용할 프롬프트 리스트
            bg_source (str): 배경 소스 ("None", "Left Light", "Right Light", "Top Light", "Bottom Light")
        """
        self.coco_json_path = Path(coco_json_path)
        self.input_img_dir = Path(input_img_dir)
        self.output_dir = Path(output_dir)
        self.prompts = prompts
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.seed = seed
        self.bg_source = bg_source
        self.gpu_id = gpu_id
        self.total_gpus = total_gpus
        self.device = device
        
        # 출력 디렉토리 생성
        self.output_images_dir = self.output_dir / "images"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 어노테이션 시각화 결과 저장 디렉토리
        self.output_anno_dir = self.output_dir / "output_result_anno"
        self.output_anno_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO json 로드
        with open(self.coco_json_path, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
            
        # 이미지 ID와 파일명 매핑
        self.image_id_to_file = {
            img['id']: img['file_name'] for img in self.coco_data['images']
        }
        
        self.ic_light = get_ic_light(device=self.device)

    def _visualize_annotation(self, image, annotations, output_path):
        """어노테이션 시각화"""
        # 이미지 복사
        vis_image = image.copy()
        
        # 각 어노테이션에 대해 바운딩 박스 그리기
        for ann in annotations:
            x, y, w, h = map(int, ann['bbox'])
            category_id = ann['category_id']
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 카테고리 ID 표시
            cv2.putText(vis_image, f"ID: {category_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 결과 저장
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    def generate(self):
        """데이터 생성 실행"""
        new_images = []
        new_annotations = []
        new_image_id = 0
        
        # GPU별로 다른 시드 사용
        gpu_seed = self.seed + (self.gpu_id * 10000)
        
        # 어노테이션이 있는 이미지 ID 목록 생성
        image_ids_with_annotations = set(ann['image_id'] for ann in self.coco_data['annotations'])
        
        # 어노테이션이 있는 이미지만 필터링
        images_to_process = [
            img for img in self.coco_data['images'] 
            if img['id'] in image_ids_with_annotations
        ]
        
        # 각 이미지에 대해 처리
        for img_info in tqdm(images_to_process, desc=f"Processing images on GPU {self.gpu_id}"):
            img_id = img_info['id']
                
            # 원본 이미지 로드
            img_path = self.input_img_dir / self.image_id_to_file[img_id]
            if not img_path.exists():
                continue
                
            try:
                input_img = Image.open(img_path).convert('RGB')
                input_array = np.array(input_img)
                original_size = (input_img.width, input_img.height)
                
                # 원본 이미지의 어노테이션 찾기
                original_annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == img_id]
                
                # 각 프롬프트에 대해 이미지 생성
                for prompt_idx, prompt in enumerate(self.prompts):
                    try:
                        # IC-Light를 사용한 이미지 생성
                        _, outputs = self.ic_light.process_relight(
                            input_fg=input_array,
                            prompt=prompt,
                            image_width=self.image_width,
                            image_height=self.image_height,
                            num_samples=self.batch_size,
                            seed=gpu_seed,  # GPU별 시드 사용
                            steps=25,
                            a_prompt="best quality",
                            n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
                            cfg=2.0,
                            highres_scale=1.5,
                            highres_denoise=0.5,
                            lowres_denoise=0.9,
                            bg_source=self.bg_source
                        )
                        
                        # 최종 이미지 크기 계산
                        final_width = int(round(self.image_width * 1.5 / 64.0) * 64)
                        final_height = int(round(self.image_height * 1.5 / 64.0) * 64)
                        
                        # 생성된 각 이미지 저장
                        for batch_idx, output in enumerate(outputs):
                            # 새 이미지 정보 생성
                            new_file_name = f"{new_image_id:06d}.png"
                            output_path = self.output_images_dir / new_file_name
                            
                            # 이미지 저장
                            Image.fromarray(output).save(output_path)
                            
                            # 이미지 정보 추가
                            new_image_info = {
                                'id': new_image_id,
                                'file_name': new_file_name,
                                'height': final_height,  # highres_scale이 적용된 크기
                                'width': final_width,    # highres_scale이 적용된 크기
                                'original_file': str(img_path),
                                'prompt': prompt
                            }
                            new_images.append(new_image_info)
                            
                            # 어노테이션 조정 및 추가
                            current_annotations = []
                            for ann in original_annotations:
                                new_ann = ann.copy()
                                new_ann['image_id'] = new_image_id
                                new_ann['id'] = f"{new_image_id}_{batch_idx}_{ann['id']}"
                                
                                # 바운딩 박스 조정
                                new_ann['bbox'] = self._adjust_bbox(
                                    ann['bbox'], 
                                    original_size, 
                                    (final_width, final_height)
                                )
                                
                                # 면적 재계산
                                new_ann['area'] = new_ann['bbox'][2] * new_ann['bbox'][3]
                                
                                new_annotations.append(new_ann)
                                current_annotations.append(new_ann)
                            
                            # 어노테이션 시각화 및 저장
                            anno_output_path = self.output_anno_dir / f"anno_{new_file_name}"
                            self._visualize_annotation(output, current_annotations, anno_output_path)
                            
                            new_image_id += 1
                            
                    except Exception as e:
                        print(f"Error processing prompt {prompt}: {e}")
                        continue
                            
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        # 새로운 COCO 포맷 데이터 생성
        new_coco_data = {
            'info': {
                'description': 'Synthetic dataset generated with IC-Light',
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'licenses': [],
            'categories': self.coco_data['categories'],  # 카테고리 정보 유지
            'images': new_images,
            'annotations': new_annotations
        }
        
        # 결과 JSON 저장
        output_json_path = self.output_dir / f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_coco_data, f, indent=2)
            
        print(f"생성 완료: {len(new_images)}개의 이미지와 {len(new_annotations)}개의 어노테이션이 생성되었습니다.")

    def _adjust_bbox(self, bbox, original_size, new_size):
        """바운딩 박스 좌표를 center crop에 맞게 조정"""
        orig_w, orig_h = original_size
        target_w, target_h = new_size
        
        # center crop을 위한 스케일과 오프셋 계산
        scale = max(target_w / orig_w, target_h / orig_h)
        scaled_w = int(round(orig_w * scale))
        scaled_h = int(round(orig_h * scale))
        
        offset_x = (scaled_w - target_w) / 2
        offset_y = (scaled_h - target_h) / 2
        
        # 원본 바운딩 박스 좌표
        x, y, w, h = bbox
        
        # 새로운 좌표 계산
        new_x = x * scale - offset_x
        new_y = y * scale - offset_y
        new_w = w * scale
        new_h = h * scale
        
        # 바운딩 박스가 이미지 범위를 벗어나는지 확인하고 조정
        if new_x < 0:
            new_w += new_x  # 너비 감소
            new_x = 0
        if new_y < 0:
            new_h += new_y  # 높이 감소
            new_y = 0
        
        # 오른쪽과 아래쪽 경계 체크
        if new_x + new_w > target_w:
            new_w = target_w - new_x
        if new_y + new_h > target_h:
            new_h = target_h - new_y
        
        # 음수 크기 방지
        new_w = max(0, new_w)
        new_h = max(0, new_h)
        
        return [new_x, new_y, new_w, new_h]