# Theo dõi lab theo Checkpoint

## Checkpoint 1 — Môi trường, dữ liệu, nhãn

Đã bổ sung mã và notebook; **chưa xác nhận chạy trên dữ liệu thật/Colab**.

Kiểm chứng cục bộ ngày 14/10/2026: `83 passed, 1 skipped` khi chạy
`.venv/bin/python -m pytest -q tests student/tests`. JSON notebook và cú pháp các cell
Python hợp lệ; `git diff --check` sạch. Không có thay đổi trong `scripts/` hoặc `tests/`.

Mã của sinh viên nằm trong `student/`. Giữ nguyên `scripts/` và `tests/` của đề.
`student/data.py` cung cấp `yolo_to_box(label, yolo)`, `objects_to_answer(objects)`,
`inspect_splits(data_dir)` và công cụ vẽ so sánh YOLO gốc với JSON quy ngược về pixel.

### Kiểm tra trên máy

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r student/requirements-local.txt
.venv/bin/python -m pytest -q tests student/tests
```

### Kiểm chứng trên Colab T4

1. Đưa mã Checkpoint 1 lên repo bằng commit/push của bạn để Colab clone được phiên bản mới.
2. Mở `lab_day10.ipynb` trên Colab, chọn T4 và chạy các cell của Checkpoint 1.
3. Thêm Secrets `KAGGLE_USERNAME` và `KAGGLE_KEY`, cấp notebook quyền truy cập. Nếu dùng zip/thư mục giải nén, chọn `DATA_MODE` tương ứng và điền đường dẫn; không cần Secrets.
4. Kiểm tra số ảnh train/val/test là 1422/276/311, số box khớp `meta.json`, không trùng session, và dataset version khớp split thực.
5. Xem đủ sáu ảnh so sánh box gốc/box JSON. Sai số lượng tử hoá tối đa là 0,32 pixel theo x và 0,18 pixel theo y với ảnh 640×360.
6. Cell cuối sao lưu vào Drive và tải `checkpoint1.zip`. Gửi zip và log pytest để kiểm chứng; dừng ở mốc Checkpoint 1.

Nếu clone Colab đã tồn tại và chưa có mã mới, chạy `git pull --ff-only` trong thư mục repo trước khi chạy lại. Mã đã import cần được tải lại bằng restart runtime nếu bạn sửa mã trên Colab.

### Nội dung commit/push

```bash
git add lab_day10.ipynb student/
git diff --cached --stat
git commit -m "feat: checkpoint 1 data validation and grounding labels"
git push
```

Không thêm `.venv/`, `data/`, `work/`, Secrets, hay file trọng số vào Git.
`checkpoint1.zip` chỉ là bằng chứng kiểm tra dữ liệu, không thay thế evidence chấm điểm của E0–E4.

## Các mốc tiếp theo

| Mốc | Kết quả cần có trước khi dừng |
|---|---|
| 2 — E0 | Suy luận thật, `e0.json`, ví dụ output và lỗi |
| 3 — E1 | Mask loss đúng, LoRA LLM, train log, `e1.json` |
| 4 — E2 | Lưu/nạp projector đầy đủ, train log, `e2.json` |
| 5 — E3 | LoRA ViT, train log, `e3.json`, so sánh E1–E3 |
| 6 — E4/báo cáo | Ablation 10%/100%, báo cáo, ảnh dự đoán, kiểm tra nộp |

Chỉ đánh dấu thí nghiệm hoàn tất khi có số liệu từ lần chạy thật. Bạn tự commit/push sau từng mốc; chưa làm E5.

Lưu ý khi hoàn thiện báo cáo: một test gốc yêu cầu còn placeholder `<điền` trong `REPORT.md`, nhưng checker nộp bài yêu cầu xoá placeholder. Khi đến Checkpoint 6, giữ nguyên test và ghi rõ xung đột; báo cáo cuối phải được điền đầy đủ.
