import json
import os
from pathlib import Path
from datetime import datetime
def update_coco_json(json_path, images_dir, output_path=None):
    """
    이미지 디렉토리에 존재하는 이미지들만 포함하도록 COCO JSON 파일을 업데이트합니다.
    이미지 경로를 파일 이름만 포함하도록 수정합니다.
    
    Args:
        json_path (str): 원본 COCO JSON 파일 경로
        images_dir (str): 이미지가 저장된 디렉토리 경로
        output_path (str, optional): 출력할 JSON 파일 경로. None이면 자동 생성됩니다.
    """
    # 경로 객체로 변환
    json_path = Path(json_path)
    images_dir = Path(images_dir)
    
    # 현재 존재하는 이미지 파일 목록 가져오기
    existing_images = set(f.name for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    print(f"디렉토리에서 발견된 이미지 수: {len(existing_images)}")
    
    # COCO JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 존재하는 이미지만 필터링하고 파일 이름만 사용하도록 수정
    new_images = []
    for img in coco_data['images']:
        # 전체 경로에서 파일 이름만 추출
        file_name = Path(img['file_name']).name
        if file_name in existing_images:
            img['file_name'] = file_name  # 파일 이름만 저장
            new_images.append(img)
            
    valid_image_ids = set(img['id'] for img in new_images)
    
    # 유효한 이미지에 대한 어노테이션만 필터링
    new_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in valid_image_ids]
    
    print(f"원본 이미지 수: {len(coco_data['images'])}")
    print(f"원본 어노테이션 수: {len(coco_data['annotations'])}")
    print(f"업데이트된 이미지 수: {len(new_images)}")
    print(f"업데이트된 어노테이션 수: {len(new_annotations)}")
    
    # 새로운 COCO 데이터 생성
    new_coco_data = {
        'info': {
            'description': 'Updated COCO dataset',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_file': str(json_path)
        },
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': new_images,
        'annotations': new_annotations
    }
    
    # 출력 경로가 지정되지 않은 경우 자동 생성
    if output_path is None:
        output_path = json_path.parent / f"updated_coco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_path = Path(output_path)
    
    # 새로운 JSON 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"업데이트된 COCO JSON 파일이 저장되었습니다: {output_path}")

if __name__ == "__main__":
    # 예시 사용법
    json_path = "output/merged_synthetic_data_20241211_055344.json"  # 원본 JSON 파일 경로
    images_dir = "output/merged_images"  # 이미지가 저장된 디렉토리 경로
    
    update_coco_json(json_path, images_dir)