# Text-based Person Search (ITSELF + Identity sampler + reranking)

Project thử nghiệm text-based person search trên RSTPReid theo hai giai đoạn: train retrieval model **ITSELF + Identity sampler** từ checkpoint RSTP CLIP, rồi train một cross-modal token matcher để rerank top ứng viên. Reranker học phân loại cặp ảnh-caption cùng identity/khác identity; encoder CLIP giữ frozen trong giai đoạn này, còn fusion Transformer và classifier được train trên RSTPReid.

Notebook Kaggle hoàn chỉnh: [`notebooks/itself-identity-reranker-rstpreid.ipynb`](notebooks/itself-identity-reranker-rstpreid.ipynb). Notebook clone implementation ITSELF từ [TBPS-Sampler](https://github.com/HoangVo-Prog/tbps-sampler), chọn `val` để chọn checkpoint, train reranker, rồi so sánh baseline/rerank trên test và xuất thống kê query bị hại. Cấu hình dùng ViT-B/16, ảnh 384×128 và direct resize đúng như ITSELF.

Lưu ý phạm vi: bước reranker hiện lấy global CLIP similarity để tạo candidate top M, sau đó dùng token matcher chấm lại. Đây chưa phải scoring `global+GRAB` đầy đủ của ITSELF; kết quả reranker cần được hiểu là thí nghiệm trên global retrieval của encoder đã train bằng ITSELF. Checkpoint, dữ liệu và output Kaggle không được commit vào Git.

## Cài đặt

Python 3.10+ được hỗ trợ. PyTorch phải được cài theo cấu hình máy (CPU hoặc CUDA); sau đó cài project:

```powershell
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e .
```

Nếu cần CUDA, cài bản PyTorch phù hợp với phiên bản CUDA trên máy theo hướng dẫn chính thức của PyTorch trước khi cài project.

## Tải model và chạy kiểm tra

```powershell
person-search download
person-search smoke
```

`download` khởi tạo model để tải/cached weights. `smoke` mã hóa hai mô tả mẫu và in kích thước vector cùng cosine similarity.

## Tìm kiếm trong gallery

Đặt ảnh gallery vào một thư mục, rồi chạy:

```powershell
person-search search `
  --query "a man wearing a red jacket, dark trousers, and carrying a backpack" `
  --images .\data\gallery `
  --top-k 10 `
  --output .\outputs\ranking.csv
```

Chương trình xếp hạng ảnh theo cosine similarity giữa embedding văn bản và embedding ảnh. Ảnh được đọc đệ quy từ thư mục gallery. Trước mắt mỗi ảnh là một đơn vị tìm kiếm; project chưa phát hiện/cắt người trong ảnh.

## Tải dữ liệu RSTPReid

Tải benchmark Kaggle về thư mục `data` (cần Kaggle CLI và đăng nhập Kaggle đã được cấu hình):

```powershell
python -m pip install kaggle
kaggle datasets download -d hoanggv/tbps-benchmark -p .\data --unzip
```

Trỏ `--root` tới thư mục có `imgs/` và `data_captions.json`, ví dụ `data\tbps-benchmark\benchmark\RSTPReid`.

## Train reranker với checkpoint đã có

Lệnh sau dùng CLIP checkpoint đã fine-tune trên RSTPReid. Reranker chỉ học trên `train`; `val` được dùng để chọn checkpoint tốt nhất. `test` không được dùng trong quá trình train:

```powershell
person-search train-reranker-rstpreid `
  --root .\data\tbps-benchmark\benchmark\RSTPReid `
  --clip-checkpoint .\models\rstp.pth `
  --clip-model ViT-B-16 `
  --image-size 384 128 `
  --output .\outputs\rstpreid-reranker.pth `
  --batch-size 32 `
  --epochs 5
```

Mỗi positive là ảnh và caption ghép trong dữ liệu. Negative là caption của một person ID khác. Image/text token features được lấy từ CLIP; hai loại token được nối với một token học được và qua hai lớp fusion Transformer, sau đó classifier dự đoán match/non-match. Reranker hiện giữ CLIP frozen để train phần fusion nhỏ hơn trên dữ liệu TBPS.

## Đánh giá baseline và reranker

Đánh giá chỉ trên test mặc định. Lệnh báo metric trước/sau rerank, chênh lệch, số query bị hại ở từng Recall@K và số query có AP thấp hơn. `--per-query-output` ghi kết quả từng caption thành CSV:

```powershell
person-search evaluate-rstpreid `
  --root .\data\tbps-benchmark\benchmark\RSTPReid `
  --clip-checkpoint .\models\rstp.pth `
  --clip-model ViT-B-16 `
  --image-size 384 128 `
  --reranker-checkpoint .\outputs\rstpreid-reranker.pth `
  --rerank-top-m 100 `
  --batch-size 32 `
  --output .\outputs\rstpreid-rerank-metrics.json `
  --per-query-output .\outputs\rstpreid-rerank-per-query.csv
```

Reranker sắp thứ tự lại top M ứng viên theo global CLIP; phần còn lại giữ nguyên thứ tự. `--rerank-top-m 0` rerank toàn bộ gallery (tốn thời gian hơn). Reranking không thể đưa ảnh đúng vào top M nếu baseline đã xếp ảnh đó ngoài pool; metrics R@K sau rerank vì vậy cần được đọc cùng baseline Recall@M.

Với checkpoint ITSELF, thêm `--image-preprocess itself` ở cả lệnh train và evaluate. Tùy chọn này resize trực tiếp ảnh về 384×128 như upstream ITSELF; `open_clip` dùng preprocessing/crop mặc định khác.

Để đo riêng baseline với đúng checkpoint, bỏ `--reranker-checkpoint`. Có thể chạy `--split val` để xem validation metrics, nhưng chỉ dùng test một lần cho báo cáo cuối. Nếu Kaggle đóng gói thư mục khác, thay đường dẫn `--root` cho phù hợp.

## Cấu trúc

```text
src/person_search/model.py       # tải CLIP checkpoint và tạo global/token features
src/person_search/reranker.py    # token fusion matcher và checkpoint loader
src/person_search/training.py    # train reranker, rerank, metric và per-query harm
src/person_search/cli.py         # train/evaluate/search commands
```

## Lộ trình person search

Reranker là cross-encoder nên chỉ nên dùng trên shortlist; CLIP global similarity xử lý truy xuất bước đầu. Kết quả còn phụ thuộc vào đúng model architecture, preprocessing và checkpoint. RSTPReid checkpoint cần được tạo từ train split, còn validation/test phải giữ riêng để có đánh giá đáng tin cậy.

