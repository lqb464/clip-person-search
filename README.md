# Text-based Person Search (pretrained ITSELF + reranking)

Pipeline dùng checkpoint có sẵn [GenAI4ELab/itself-rstp](https://huggingface.co/GenAI4ELab/itself-rstp), fine-tune cả encoder ảnh/văn bản và một cross-modal token matcher để rerank top 100 ứng viên trên RSTPReid. Khởi tạo trực tiếp từ checkpoint pretrained; không cần đính kèm model Kaggle `rstp.pth`.

Notebook Kaggle: [`notebooks/itself-identity-reranker-rstpreid.ipynb`](notebooks/itself-identity-reranker-rstpreid.ipynb). Tên file được giữ để các link cũ tiếp tục hoạt động. Đính kèm dataset `tbps-benchmark`, bật Internet và GPU, rồi chạy từ đầu trên phiên mới. Notebook tải `best.pth` (khoảng 1,91 GB), khóa revision `d575c9691832b0a3e88d076faeba9973260956ee`, kiểm tra nạp encoder và trích token trước khi train reranker. Encoder và reranker được train chung 5 epoch (LR encoder `1e-6`, LR matcher `2e-4`, batch size 4 trong notebook); validation loss chọn checkpoint, test dùng để đo baseline/rerank và thống kê query bị hại.

Repo checkpoint không cung cấp config hoặc thông tin sampler, nên chưa xác nhận checkpoint được train bằng Identity sampler. Notebook dùng ViT-B/16, ảnh 384×128; bước kiểm tra sẽ dừng nếu checkpoint không khớp. Thí nghiệm hiện dùng global CLIP similarity của encoder ITSELF để tạo candidate, chưa dùng scoring global+GRAB đầy đủ. Không commit checkpoint, dữ liệu hoặc output vào Git.

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
  --batch-size 4 `
  --epochs 5
```

Mỗi positive là ảnh và caption ghép trong dữ liệu. Negative là caption của một person ID khác. Image/text token features được lấy từ CLIP; hai loại token được nối với một token học được và qua hai lớp fusion Transformer, sau đó classifier dự đoán match/non-match. Gradient đi qua token features để cập nhật cả encoder CLIP và matcher. Checkpoint tốt nhất lưu `encoder_state_dict` cùng trọng số matcher; evaluator tự khôi phục encoder đó. Checkpoint reranker frozen cũ vẫn đọc được.

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


## Fine-tune encoder cùng reranker

Encoder ảnh/văn bản hiện được mở gradient mặc định. `--encoder-learning-rate` mặc định `1e-6`; `--learning-rate` cho matcher mặc định `2e-4`. Fine-tune dùng thêm VRAM, nên notebook giảm batch size train xuống 4 (có thể giảm tiếp xuống 2).

Checkpoint mới chứa cả encoder và matcher tại cùng epoch được chọn. Khi đánh giá có `--reranker-checkpoint`, cả baseline và reranked dùng encoder từ checkpoint này. Notebook còn xuất `rstpreid-original-itself-metrics.json` để so với encoder Hugging Face gốc, giúp phân biệt ảnh hưởng của fine-tune encoder và ảnh hưởng của rerank. Test không tham gia chọn checkpoint.
