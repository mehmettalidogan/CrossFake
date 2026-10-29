# Dataset download and layout

Do **not** scrape unofficial mirrors. Request / download each dataset from its authors, then point the extractor at the raw root.

After extraction the project expects:

```
data/faces/manifest.csv
data/faces/{ffpp|celebdf|dfdcp}/{train|val|test}/{video_id}/{frame_idx:06d}.jpg
data/faces/{ffpp|celebdf|dfdcp}/{train|val|test}/{video_id}/{frame_idx:06d}.json
```

`manifest.csv` columns: `path,label,source_dataset,split,video_id,manip_type,frame_idx`  
`label`: `0` real, `1` fake.

FF++ official split JSON files (video IDs only, not media) are fetched automatically into `data/raw/ffpp/splits/`.

## FaceForensics++ (train source, c23 / HQ)

1. Fill the request form: https://github.com/ondyari/FaceForensics
2. Download **c23** originals + four manipulations: Deepfakes, Face2Face, FaceSwap, NeuralTextures.
3. Layout:

```
data/raw/ffpp/
  original_sequences/youtube/c23/videos/*.mp4
  manipulated_sequences/Deepfakes/c23/videos/*.mp4
  manipulated_sequences/Face2Face/c23/videos/*.mp4
  manipulated_sequences/FaceSwap/c23/videos/*.mp4
  manipulated_sequences/NeuralTextures/c23/videos/*.mp4
```

```powershell
python src/data/extract_faces.py --dataset ffpp --input data/raw/ffpp --compression c23 --num-frames 32
```

## Celeb-DF-v2 (cross-dataset test)

https://github.com/yuezunli/celeb-deepfakeforensics

```
data/raw/celebdf/
  Celeb-real/
  YouTube-real/
  Celeb-synthesis/
  List_of_testing_videos.txt
```

If `List_of_testing_videos.txt` is present, only those videos are written as `split=test` (standard CDF protocol). Other videos are skipped (`unused`). If the list is missing, every video is tagged `test`.

```powershell
python src/data/extract_faces.py --dataset celebdf --input data/raw/celebdf --num-frames 32
```

## DFDC Preview (DFDCP, cross-dataset test)

https://ai.meta.com/datasets/dfdc/ (preview / smaller pack, not the full 470 GB set)

Supported metadata: `metadata.json` (Kaggle-style) or `labels.csv` with `filename,label`.

```powershell
python src/data/extract_faces.py --dataset dfdcp --input data/raw/dfdcp --num-frames 32
```

## Disk note

Raw FF++ c23 is large. Face JPEG crops (32 frames/video) are much smaller than keeping every mp4, but still tens of GB for the full protocol. Use `--limit N` on the extractor while debugging.
