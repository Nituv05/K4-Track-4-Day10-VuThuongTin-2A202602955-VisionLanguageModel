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

## Checkpoint 2 — E0 zero-shot

Đã triển khai mã E0; **chưa có kết quả GPU/dữ liệu thật để xác nhận E0 hoàn tất**.
Kiểm chứng cục bộ ngày 14/10/2026: **93 passed, 1 skipped**, bao gồm kiểm tra
cắt prompt khi batch có left padding, tiếp tục sau lỗi giữa chừng, bảo toàn output thô,
phát hiện sai cấu hình resume và chấm bằng scorer gốc trên dữ liệu giả lập.
Các test tensor chạy CPU với torch; nếu môi trường cục bộ chưa có torch, nhóm này được skip.
Colab đã có torch nên phải chạy được nhóm test này. JSON notebook, cú pháp cell và `git diff --check` hợp lệ.
`student/inference.py` nạp NF4 4-bit, giữ vision và lm_head ngoài lượng tử hoá,
kiểm tra cây module/tham số và 220 token ảnh, rồi suy luận batch với left padding.
Đầu ra giữ nguyên chuỗi model trả về, không sửa JSON trước khi chấm.

### Chạy trên Colab

1. Tự commit/push mã Checkpoint 2. Clone mới hoặc `git pull --ff-only` để lấy phiên bản đó; restart runtime nếu đã import mã cũ.
2. Chạy phần môi trường/dữ liệu của Checkpoint 1 nếu dùng runtime mới. Cần dữ liệu đã chuẩn bị và Drive đã mount.
3. Chạy phần **Checkpoint 2** trong mục `## 3. E0`: nạp model rồi chạy 40 ảnh val đầu tiên.
4. Đọc năm output thô và bộ đếm triệu chứng. Đối chiếu ảnh/GT để phân biệt JSON hỏng, sai lớp, sai thang và bịa box. Bộ đếm tự động chỉ phản ánh triệu chứng.
5. Sau khi kiểm tra val, chạy toàn bộ 311 ảnh test. Scorer gốc tạo `report/evidence/e0.json` sau khi xác nhận đủ ID test.
6. Tải `checkpoint2.zip`, gửi zip và log để kiểm chứng; đưa riêng `e0.json` đã chấm thật về thư mục `report/evidence/` trong repo của bạn.

Output và `.run.json` được sao lưu Drive sau mỗi batch. Khi Colab ngắt, chạy lại cùng nhãn,
prompt, model revision và cấu hình; các ảnh đã lưu không bị suy luận lại.
Nếu cấu hình thay đổi, runner từ chối trộn kết quả; chọn `E0_RUN_LABEL` mới.
Nếu OOM, restart runtime, đặt batch 1 và nhãn `e0_b1`, kiểm tra trên val trước khi chạy test.
Không chỉnh cấu hình/prompt theo điểm test. Không chấm 40 ảnh val như bằng chứng test.

### Kiểm tra và commit

```bash
.venv/bin/python -m pytest -q tests student/tests
git add lab_day10.ipynb student/
# Chỉ chạy dòng này sau khi lấy file chấm thật từ Colab:
git add report/evidence/e0.json
git diff --cached --stat
git commit -m "feat: checkpoint 2 zero-shot inference and scoring"
git push
```

Mã E0 cần được push trước để Colab lấy được; có thể commit bằng chứng thật riêng sau khi chạy.
Tiêu chí hoàn tất mốc: đủ 311 dự đoán test, `missing_predictions = 0`, `split = test`,
dataset version khớp split chuẩn, có parse valid rate/F1/CI95 thật và năm output thô.
E0 có thể có chất lượng thấp; ngưỡng parse 95% và tăng F1 0,10 áp dụng cho E1.

Tham khảo API chính thức:
[Qwen3-VL](https://huggingface.co/docs/transformers/model_doc/qwen3_vl),
[bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

Lưu ý khi hoàn thiện báo cáo: một test gốc yêu cầu còn placeholder `<điền` trong `REPORT.md`, nhưng checker nộp bài yêu cầu xoá placeholder. Khi đến Checkpoint 6, giữ nguyên test và ghi rõ xung đột; báo cáo cuối phải được điền đầy đủ.
