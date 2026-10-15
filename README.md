# Text-based Person Search (CLIP + reranking)

Project cho bài toán tìm ảnh người từ mô tả tự nhiên trên RSTPReid. Pipeline gồm CLIP fine-tuned để lấy top ứng viên và một cross-modal token matcher để rerank top 100. Matcher học phân loại cặp ảnh-caption khớp/không khớp; backbone CLIP được giữ frozen trong giai đoạn đầu, còn fusion Transformer và classifier được train trên RSTPReid.

Model chung mặc định là OpenCLIP `ViT-B-32`. Với checkpoint TBPS RSTPReid được cung cấp, dùng `ViT-B-16` và kích thước ảnh `384 128` để khớp patch embedding và positional embedding của checkpoint. Checkpoint CLIP phải tương thích với kiến trúc và kích thước ảnh đã chọn. Đặt `rstp.pth` tại `models/rstp.pth` hoặc truyền đường dẫn trực tiếp. Không commit checkpoint, dữ liệu hay output vào Git.

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

Trỏ `--root` tới thư mục có `imgs/` và `data_captions.json`, ví dụ `.data	bps-benchmarkenchmarkRSTPReid`.

## Train reranker

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

Reranker sắp thứ tự lại top M ứng viên theo CLIP; phần còn lại giữ nguyên thứ tự CLIP. `--rerank-top-m 0` rerank toàn bộ gallery (tốn thời gian hơn). Reranking không thể đưa ảnh đúng vào top M nếu CLIP đã xếp ảnh đó ngoài pool; metrics R@K sau rerank vì vậy cần được đọc cùng baseline Recall@M.

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

