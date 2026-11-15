# Task B — Hướng dẫn chi tiết về cột `freshness` (tiếng Việt)

Tài liệu này làm rõ **duy nhất một cột**: cột `freshness` trong file
`task_b_evidence.csv`. Phần còn lại của Task B (cột `supports`) và toàn bộ
Task A vẫn theo `GUIDELINES.md`.

> **Lưu ý bắt buộc:** dù hướng dẫn viết bằng tiếng Việt, **giá trị điền vào ô
> phải gõ bằng tiếng Anh, viết HOA**: `CURRENT`, `SUPERSEDED`, `ANACHRONISTIC`,
> `UNDATED`, `NOT_APPLICABLE`. Script chấm điểm so khớp đúng các chuỗi này; gõ
> tiếng Việt hoặc viết thường sẽ bị coi là nhãn không hợp lệ.

---

## 1. Câu hỏi mà cột này thực sự hỏi

> *Tại thời điểm `question_date`, nội dung của đoạn trích này có mô tả đúng
> tình trạng lúc đó không?*

Ba điểm dễ hiểu sai nhất, nói thẳng ra trước:

**(a) Mốc so sánh là `question_date`, KHÔNG phải hôm nay.**
Đây là lỗi phổ biến nhất của cả Task A lẫn Task B. Một tài liệu nói đúng
sự thật *bây giờ* nhưng sai *vào ngày đặt câu hỏi* thì không phải `CURRENT`.

**(b) Hãy xét MỐC THỜI GIAN CỦA NỘI DUNG, không phải cột `evidence_date`.**
Cột `evidence_date` là ngày xuất bản, chỉ mang tính gợi ý. Hai trường hợp thường
gặp khiến ngày xuất bản đánh lừa:

- Một bài **hồi cố** đăng năm 2026 có thể mô tả chính xác tình hình tháng 6/2022.
  Với câu hỏi ngày 2022-06-16, nội dung đó là `CURRENT`.
- Một bài đăng **trước** ngày đặt câu hỏi vẫn có thể đã lỗi thời, nếu sự việc đã
  thay đổi trong khoảng giữa.

**(c) `freshness` được chấm ĐỘC LẬP với `supports`.**
Đừng để hai cột ảnh hưởng nhau. Mọi tổ hợp đều có thật: một tài liệu có thể
`CURRENT` mà không chứa đáp án, hoặc `SUPERSEDED` mà vẫn chứa đúng chuỗi đáp án
(đây chính là kiểu lỗi mà nghiên cứu này muốn đo).

---

## 2. Cây quyết định

Làm theo đúng thứ tự này:

```
Bước 1 — Sự việc này có thể thay đổi theo thời gian không?
   KHÔNG  → NOT_APPLICABLE   (định nghĩa, hằng số vật lý, sự kiện lịch sử đã chốt)
   CÓ     → sang Bước 2

Bước 2 — Nội dung mô tả thời điểm nào, so với question_date?
   Đúng tình trạng TẠI question_date        → CURRENT
   Tình trạng CŨ HƠN, đã bị thay thế        → SUPERSEDED
   Tình trạng MUỘN HƠN question_date        → ANACHRONISTIC
   Không xác định được thời điểm nào cả     → UNDATED
```

`SUPERSEDED` và `ANACHRONISTIC` **ngược chiều nhau**. Nhầm lẫn hai nhãn này làm
hỏng đúng phép đo mà nghiên cứu cần, nên hãy đọc kỹ mục 3.

---

## 3. Giải thích từng nhãn

### `CURRENT`
Nội dung mô tả đúng tình trạng tại `question_date`.
Ngày xuất bản có thể trước, bằng, hoặc thậm chí sau `question_date` — miễn là
*nội dung* nói về tình trạng đúng vào lúc đó.

### `SUPERSEDED` — tài liệu **cũ** đã bị thay thế
Nội dung từng đúng, nhưng đến `question_date` thì đã bị sự kiện sau đó vượt qua.

> Câu hỏi (2023-02-24): *Ai là Chánh văn phòng Nhà Trắng?* — Đáp án: `Jeff Zients`
> Đoạn trích: *"Ron Klain giữ chức Chánh văn phòng Nhà Trắng từ năm 2021…"*
> → `SUPERSEDED`. Đúng vào năm 2021–2022, nhưng sai vào 24/02/2023.

Đây là trường hợp **bằng chứng lỗi thời** — trọng tâm của cả bài báo.

### `ANACHRONISTIC` — tài liệu **mới hơn** câu hỏi

> **Nhãn này mới được bổ sung.** Khoảng **1/9 số dòng** trong Task B có bằng
> chứng xuất bản *sau* ngày đặt câu hỏi, có trường hợp sau tới vài năm. Nguyên
> nhân: kho tài liệu gộp bằng chứng của *tất cả* câu hỏi và **không** lọc theo
> mốc thời gian riêng của từng câu. Đây là đặc tính thật của hệ thống đang được
> kiểm định, không phải lỗi dữ liệu — và việc gán nhãn đúng chính là một phần
> mục tiêu của nghiên cứu.

Dùng khi nội dung mô tả tình trạng **muộn hơn** `question_date`: thông tin chưa
tồn tại vào lúc câu hỏi được đặt ra.

> Câu hỏi (2022-09-02): *Tên lửa Artemis I được đưa về đâu?* — Đáp án: `Vehicle Assembly Building`
> Đoạn trích: *"25/02/2026 — NASA đang đưa tên lửa SLS từ bệ phóng về Vehicle
> Assembly Building để sửa chữa."*
> → `supports = YES` (có nêu đúng đáp án) nhưng `freshness = ANACHRONISTIC`:
> nội dung thuộc năm 2026, không phải bằng chứng có được vào 09/2022.

**Phân biệt quan trọng:** nếu bài đăng năm 2026 nhưng *nội dung* kể lại tình
hình tại `question_date` (bài hồi cố), thì đó là `CURRENT`, **không** phải
`ANACHRONISTIC`. Hãy xét nội dung, đừng xét ngày đăng.

### `UNDATED`
Không thể định vị nội dung trong thời gian, và cột ngày cũng không giúp được.
Dùng khi thực sự bế tắc, không dùng để né quyết định.

### `NOT_APPLICABLE`
Nội dung **không thể** cũ đi: định nghĩa, hằng số, sự kiện lịch sử đã cố định.

> *"Một ion có nhiều electron hơn proton thì mang điện tích âm."* → `NOT_APPLICABLE`

**Ranh giới hay bị nhầm nhất — `SUPERSEDED` và `NOT_APPLICABLE`:**

| Sự việc | Nhãn |
|---|---|
| Không thể thay đổi | `NOT_APPLICABLE` |
| Có thể thay đổi, và đã thay đổi trước `question_date` | `SUPERSEDED` |
| Có thể thay đổi, nhưng chưa thay đổi | `CURRENT` |

---

## 4. Quy tắc phụ

- **Chỉ chấm đoạn trích được hiển thị.** Đoạn trích bị cắt ở 700 ký tự. Nếu
  thông tin có thể nằm ngay sau chỗ cắt, vẫn coi như không có — chúng tôi đo
  đúng những gì mô hình được đọc.
- **Mỗi dòng là một tài liệu.** Một câu hỏi có thể trải trên tối đa ba dòng cùng
  `item_id`; chấm từng dòng riêng biệt, không gộp.
- **Không tra cứu bên ngoài.** Chỉ dựa vào nội dung trên dòng đó. Nếu buộc phải
  tra Google mới quyết được, đó là tín hiệu nên dùng `UNDATED`.
- **Ghi chú khi phân vân.** Cột `notes` nhận tiếng Việt thoải mái. Nếu bạn thấy
  một dòng nằm giữa hai nhãn, hãy chọn nhãn hợp lý nhất *và* ghi lại lý do —
  phần ghi chú này được dùng để chỉnh lại hướng dẫn.

---

## 5. Bảng ví dụ tổng hợp

| Câu hỏi (ngày) | Đáp án tham chiếu | Đoạn trích (ngày đăng) | `supports` | `freshness` | Vì sao |
|---|---|---|---|---|---|
| Ai là Chánh văn phòng Nhà Trắng? (2023-02-24) | `Jeff Zients` | "Biden bổ nhiệm Jeff Zients…" (2023-02-24) | `YES` | `CURRENT` | Đúng tại ngày hỏi |
| Ai là Chánh văn phòng Nhà Trắng? (2023-02-24) | `Jeff Zients` | "Ron Klain giữ chức từ 2021…" (2021-01-20) | `NO` | `SUPERSEDED` | Đã bị thay thế |
| Artemis I được đưa về đâu? (2022-09-02) | `Vehicle Assembly Building` | "25/02/2026 — NASA đưa SLS về VAB…" (2026-02-25) | `YES` | `ANACHRONISTIC` | Nội dung năm 2026 |
| Phim nào được duyệt mùa mới? (2022-06-16) | `"Squid Game"` | "Nhìn lại: tháng 6/2022 Netflix xác nhận mùa 2…" (2026-01-10) | `YES` | `CURRENT` | Hồi cố, nội dung đúng mốc 2022 |
| Điện tích của ion? (2022-06-17) | `negatively charged` | "Ion nhiều electron hơn proton mang điện âm." | `YES` | `NOT_APPLICABLE` | Không thể cũ đi |
| Nước nào đăng cai hội nghị? (2026-07-10) | `Turkey` | "Lãnh đạo NATO họp tại Ankara tuần này…" (2026-07-09) | `YES` | `CURRENT` | Đúng thời điểm |

---

## 6. Tự kiểm tra trước khi bắt đầu

Gán nhãn thử ba dòng sau; đáp án ở cuối.

1. Câu hỏi 2024-03-08. Đoạn trích đăng 2024-03-07, mô tả kết quả bỏ phiếu ngày
   06/03/2024 vẫn còn hiệu lực.
2. Câu hỏi 2022-11-01. Đoạn trích đăng 2025-06-01, mô tả *tình hình hiện tại năm
   2025* của cùng chủ đề.
3. Câu hỏi 2023-05-20. Đoạn trích không ghi ngày, nội dung là định nghĩa thuật
   ngữ kinh tế.

<details>
<summary>Đáp án</summary>

1. `CURRENT` — 2. `ANACHRONISTIC` (nội dung thuộc 2025, sau ngày hỏi 2022) —
3. `NOT_APPLICABLE` (định nghĩa, không cần dùng `UNDATED` dù thiếu ngày).

</details>

Nếu bạn sai từ 2 câu trở lên, hãy đọc lại mục 2 và 3 trước khi vào việc.
