

### futures/script 파일 명세
- `sampling_vcdb_frames.py`
  - `./data.json` 파일에 명세한 비디오 목록참조하여 비디오당 n개의 프레임 경로 샘플링

- `pca_vcdb_features.py`
  - 이미지에서 프레임 feature 추출후 PCA parameter 학습

- `extract_features_from_images.py`
  - 이미지에서 프레임 feature 추출 

- `extract_features_from_videos.py`
  - 비디오에서 clip feature 추출