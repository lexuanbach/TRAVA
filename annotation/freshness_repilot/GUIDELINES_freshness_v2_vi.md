# Freshness phiên bản 2 — quy tắc quyết định cụ thể hoá (tiếng Việt)

Tài liệu này thay thế phần `freshness` trong `../GUIDELINES_freshness_vi.md`.
Các phần khác của Task B giữ nguyên.

> **Bắt buộc:** giá trị điền vào ô vẫn gõ **tiếng Anh, viết HOA** —
> `CURRENT`, `SUPERSEDED`, `ANACHRONISTIC`, `UNDATED`, `NOT_APPLICABLE`,
> `NOT_ABOUT_THIS_QUESTION`, và `STATE` / `EVENT` / `TIMELESS` cho cột
> `claim_type`. Script chấm điểm so khớp đúng các chuỗi này.

**Vì sao có phiên bản 2.** Ba người gán nhãn `freshness` và chỉ đạt
Krippendorff α = 0.47, dưới ngưỡng 0.6 cần thiết. Chúng tôi đã tìm nguyên nhân
thay vì làm lại. Có hai nguyên nhân, và **không nguyên nhân nào là lỗi của người
gán nhãn**.

---

## Sửa lỗi 1 — chỉ hỏi về bằng chứng đúng chủ đề

Ở lần trước, **64% số dòng** hỏi một tài liệu có "còn đúng tại ngày đặt câu hỏi"
hay không, trong khi tài liệu đó **không liên quan gì** đến câu hỏi — ví dụ một
bài về Nichelle Nichols được ghép với câu hỏi về Cindy Williams. Câu hỏi đó
không có đáp án đúng: α chỉ đạt 0.29 ở nhóm này, so với 0.57 ở nhóm bằng chứng
thực sự liên quan.

Gói này chỉ chứa dòng đúng chủ đề. Nếu vẫn lọt dòng nào, hãy dùng
`NOT_ABOUT_THIS_QUESTION`. **Cứ dùng thoải mái** — đó là một câu trả lời thật,
không phải sự né tránh.

## Sửa lỗi 2 — quy tắc không còn hỏi bạn về thế giới

Phiên bản 1 hỏi "sự việc này có thể thay đổi không?". Đó là câu hỏi về thế giới,
và những người hợp lý trả lời khác nhau. Phiên bản 2 hỏi về **chính câu văn bạn
đang đọc**, thứ bạn có thể kiểm chứng.

---

## Bước 1 — tìm câu văn liên quan

Đọc đoạn trích và tìm **câu có liên quan đến câu hỏi**. Nếu có nhiều câu, lấy
câu gần với việc trả lời nhất. Bạn đang gán nhãn cho **câu đó**, không phải cho
cả bài báo.

**Mọi đoạn trích trong gói này đều kết thúc ở một câu hoàn chỉnh.** Lần trước thì
không: 98% bị cắt cứng theo số ký tự và 77% đứt giữa chừng một từ — mà bạn không
thể xác định thì của nửa câu. Ví dụ "…nhà máy mở năm 2019 và đang" trông như một
tình trạng đang diễn ra trong khi mệnh đề thật là một sự kiện đã hoàn tất. Lỗi
này đã được sửa: đoạn trích được cắt lùi về câu hoàn chỉnh cuối cùng, và vẫn
luôn là **một phần** của những gì mô hình đã đọc.

Nếu **không câu nào trong đoạn trích liên quan đến câu hỏi**, hãy chọn
`NOT_ABOUT_THIS_QUESTION`. Đừng suy đoán nội dung của câu bị cắt mất.

### "Liên quan đến câu hỏi" nghĩa là gì — hãy áp dụng phép thử, đừng tự phán đoán

Ở lần thử nghiệm trước, quy tắc này đạt α = 0.85 trên những dòng mà cả ba người
đều công nhận là đúng chủ đề, nhưng lại đổ vỡ đúng ở quyết định này: hai người
đánh dấu 37 và 42 trong 150 dòng là lạc đề, người thứ ba chỉ đánh dấu 1 dòng.
Cả hai cách hiểu đều hợp lý theo lời dặn "cứ dùng thoải mái" — nên nay nó có
phép thử cụ thể.

Chọn `NOT_ABOUT_THIS_QUESTION` khi **cả hai** điều sau đều đúng:

1. **Không câu nào nhắc tới đối tượng mà câu hỏi hỏi đến** — thực thể, sự kiện,
   con số hoặc chức danh trong câu hỏi không xuất hiện, dù bằng tên gọi khác; và
2. **bạn không thể đoán được câu hỏi là gì** nếu chỉ nhìn đoạn trích.

Nếu đoạn trích đúng đối tượng nhưng **không chứa đáp án**, thì vẫn là **đúng chủ
đề** — hãy sang Bước 2. "Không chứa đáp án" là việc của cột `supports`.

| Câu hỏi | Đoạn trích | Kết luận |
|---|---|---|
| Nữ diễn viên nào qua đời? (Cindy Williams) | bài về Nichelle Nichols | `NOT_ABOUT_THIS_QUESTION` |
| Đội nào thắng Super Bowl? | bài về tổng số lần vô địch | **đúng chủ đề** |
| Chiến tranh tốn bao nhiêu? | bài về cuộc chiến, không có con số | **đúng chủ đề** |

Khi phân vân, **hãy chọn đúng chủ đề**. Một nhãn freshness sai còn sửa được;
một dòng bị loại khỏi phân tích thì không.

## Bước 2 — phân loại mệnh đề (cột `claim_type`)

| `claim_type` | Câu văn khẳng định… | Ví dụ |
|---|---|---|
| `STATE` | một **tình trạng đang diễn ra**, có thể chấm dứt về sau | "Ron Klain **đang giữ** chức từ 2021", "quỹ **trị giá** 2 nghìn tỷ" |
| `EVENT` | một **sự kiện đã hoàn tất gắn với một mốc thời gian** — đã đúng thì đúng mãi | "Steelers **đã thắng** Super Bowl 2009", "tên lửa **được đưa về** ngày 2/9" |
| `TIMELESS` | định nghĩa, quy luật vật lý/toán học, thuộc tính không gắn thời gian | "ion nhiều electron hơn proton mang điện âm" |

**Dấu hiệu ngữ pháp là đáng tin, hãy dùng nó.** Thì hiện tại, "đang", "kể từ",
"tính đến nay" → `STATE`. Quá khứ kèm mốc thời gian → `EVENT`.

**Bẫy:** một bài **mới đăng** viết về một sự kiện **đã hoàn tất** vẫn là `EVENT`.
Ngày đăng không quyết định loại mệnh đề.

**Số liệu thống kê và kỷ lục là `STATE`**, không phải `EVENT`: "Steelers đã thắng
sáu lần" là con số đang tích luỹ, một lần thắng nữa sẽ thay đổi nó. Đây chính là
trường hợp gây bất đồng nhiều nhất lần trước, nay đã được quy định rõ.

## Bước 3 — gán `freshness`

| Nếu `claim_type` là… | thì `freshness` là… |
|---|---|
| `TIMELESS` | `NOT_APPLICABLE`. Luôn luôn. Dừng. |
| `EVENT` | `NOT_APPLICABLE`, **trừ khi** sự kiện xảy ra sau `question_date` → `ANACHRONISTIC`. Một sự kiện đã hoàn tất không thể bị "lỗi thời". |
| `STATE` | chọn một trong bốn nhãn dưới |

Với mệnh đề `STATE`, so tình trạng đó với `question_date`:

| `freshness` | Dùng khi |
|---|---|
| `CURRENT` | tình trạng đó vẫn đúng tại `question_date` |
| `SUPERSEDED` | tình trạng đó đã bị thay thế trước `question_date` |
| `ANACHRONISTIC` | tình trạng đó thuộc thời điểm **sau** `question_date` |
| `UNDATED` | hoàn toàn không định vị được thời gian |

**`SUPERSEDED` nay chỉ dùng cho `STATE`.** Nếu bạn định chọn `SUPERSEDED` cho một
`EVENT`, hãy đọc lại Bước 2 — đó chính là lỗi mà bản sửa đổi này nhắm tới.

---

## Ví dụ

| Câu hỏi (ngày) | Câu văn liên quan | `claim_type` | `freshness` |
|---|---|---|---|
| Ai là Chánh văn phòng? (2023-02-24) | "Ron Klain giữ chức từ 2021" | `STATE` | `SUPERSEDED` |
| Ai là Chánh văn phòng? (2023-02-24) | "Biden bổ nhiệm Jeff Zients" | `EVENT` | `NOT_APPLICABLE` |
| Đội nào thắng Super Bowl? (2026-02-12) | "Steelers và Patriots mỗi đội thắng sáu lần" | `STATE` | `SUPERSEDED` |
| Tên lửa được đưa về đâu? (2022-09-02) | "NASA đang đưa SLS về VAB" (2026) | `STATE` | `ANACHRONISTIC` |
| Nữ diễn viên nào qua đời? (2023-02-02) | "Nichelle Nichols, nổi tiếng với…" | — | `NOT_ABOUT_THIS_QUESTION` |
| Điện tích của ion? (2022-06-17) | "nhiều electron hơn proton → điện âm" | `TIMELESS` | `NOT_APPLICABLE` |

---

## Tự kiểm tra trước khi bắt đầu

1. Câu hỏi (2025-03-01). Câu: "Công ty **là** nơi tuyển dụng lớn nhất bang."
2. Câu hỏi (2025-03-01). Câu: "Công ty **đã mở** nhà máy thứ ba vào 6/2019."
3. Câu hỏi (2023-08-14). Câu: "Bà **đã giữ** ghế này từ 2019." (bà mất ghế năm 2022)
4. Câu hỏi (2022-11-01). Câu: "Tính đến 4/2025, con số **là** 12."
5. Câu hỏi (2024-01-05). Câu: "Nước **sôi** ở 100 °C tại mực nước biển."

<details><summary>Đáp án</summary>

1. `STATE` → `CURRENT` — 2. `EVENT` → `NOT_APPLICABLE` — 3. `STATE` →
`SUPERSEDED` — 4. `STATE` → `ANACHRONISTIC` — 5. `TIMELESS` → `NOT_APPLICABLE`

</details>

Sai từ 2 câu trở lên: đọc lại Bước 2 và 3 trước khi làm.

---

## Lưu ý khi điền

- Cột `notes` viết tiếng Việt thoải mái. Hãy ghi lại mọi chỗ bạn thấy quy tắc
  gượng ép — đó là cơ sở để viết phiên bản 3.
- Không trao đổi với người gán nhãn khác cho tới khi chấm xong.
- **Tránh mở CSV bằng Excel.** Gói trả về lần trước bị Excel đổi toàn bộ ngày từ
  ISO sang định dạng Mỹ. Hãy dùng trình soạn thảo văn bản, hoặc nhập mọi cột ở
  dạng Text.
