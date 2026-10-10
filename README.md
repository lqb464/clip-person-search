# Text-based Person Search (CLIP baseline)

Project khởi đầu cho bài toán tìm ảnh người từ mô tả ngôn ngữ tự nhiên. Baseline dùng OpenCLIP `ViT-B-32` với pretrained weights `laion2b_s34b_b79k`. Ở lần chạy đầu, OpenCLIP tự tải weights vào cache của Hugging Face; các lần sau dùng lại cache.

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

## Đánh giá trên RSTPReid

Tải benchmark Kaggle về thư mục `data` (cần Kaggle CLI và đăng nhập Kaggle đã được cấu hình):

```powershell
python -m pip install kaggle
kaggle datasets download -d hoanggv/tbps-benchmark -p .\data --unzip
```

Lệnh đánh giá cần thư mục RSTPReid có cấu trúc `RSTPReid\imgs\` và `RSTPReid\data_captions.json`. Sau khi giải nén, trỏ `--root` tới đúng thư mục chứa hai mục đó:

```powershell
person-search evaluate-rstpreid --root .\data\tbps-benchmark\RSTPReid --batch-size 32 --output .\outputs\rstpreid-metrics.json
```

Lệnh chỉ đánh giá các record có `split: "test"`, mã hóa toàn bộ ảnh truy vấn và gallery, rồi báo Rank/Recall@1, @5, @10 và mAP. Nếu Kaggle đóng gói thư mục khác, thay đường dẫn `--root` cho phù hợp. Không đưa dữ liệu benchmark vào Git.

## Cấu trúc

```text
src/person_search/model.py   # tải OpenCLIP và tạo embedding đã chuẩn hóa
src/person_search/cli.py     # download, smoke test, xếp hạng gallery
```

## Lộ trình person search

CLIP tổng quát là baseline để xác nhận luồng text-image, nhưng similarity toàn ảnh không đồng nghĩa với nhận dạng/tìm đúng người. Bước tiếp theo nên thêm person detector để crop từng người (một ảnh có thể sinh nhiều candidate), lập index vector cho từng crop, và đánh giá trên benchmark như CUHK-PEDES/I2T hoặc RSTPReid. Sau đó mới so sánh zero-shot với fine-tuning hoặc retrieval model chuyên biệt. Cần kiểm tra điều kiện sử dụng và quyền riêng tư của dữ liệu trước khi xây gallery thực tế.

