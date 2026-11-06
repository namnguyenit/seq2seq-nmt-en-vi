# English-to-Vietnamese Neural Machine Translation (Seq2Seq with Bahdanau Attention)

Hệ thống Dịch máy Nơ-ron (Neural Machine Translation — NMT) dịch thuật ngữ nghĩa từ **tiếng Anh sang tiếng Việt**, được xây dựng và huấn luyện bằng **PyTorch** với kiến trúc mạng tuần hoàn (Recurrent Neural Network - LSTM) kết hợp cơ chế chú ý (Bahdanau Additive Attention).

---

## 📌 Kiến trúc & Mô hình Nghiên cứu

Dự án triển khai và đối chứng thực nghiệm giữa 2 kiến trúc mạng dịch máy kinh điển:

1. **Mô hình 1: Vanilla Seq2Seq (Encoder - Decoder Không có Attention)**:
   - **Encoder**: Mạng LSTM đa tầng mã hóa toàn bộ câu nguồn thành một vector ngữ cảnh (Context Vector) kích thước cố định.
   - **Decoder**: Mạng LSTM giải mã vector ngữ cảnh thành từng từ tiếng Việt theo cơ chế tuần tự.
   - *Hạn chế*: Gặp hiện tượng nghẽn cổ chai thông tin (Information Bottleneck) khi dịch các câu có độ dài lớn (> 20 tokens).

2. **Mô hình 2: Attention-based Seq2Seq (Có Bahdanau Additive Attention)**:
   - Tích hợp lớp Attention tính toán điểm tương đồng (Alignment Score) giữa trạng thái ẩn của Decoder và tất cả các trạng thái ẩn của Encoder.
   - Cho phép Decoder "tập trung" vào các từ ngữ tương ứng trong câu gốc tiếng Anh tại mỗi bước dịch.
   - Nâng cao vượt bậc chất lượng dịch thuật tự nhiên và bảo toàn cấu trúc ngữ pháp tiếng Việt.

---

## 📂 Cấu trúc Dự án

```text
seq2seq-nmt-en-vi/
├── MachineTranslation_en_vi/
│   ├── .gitattributes
│   └── README.md
├── add_eval_demo.py         # Script bổ sung ô demo suy luận vào Notebook
├── best_model_no_attn.pt    # Checkpoint trọng số tối ưu (Mô hình 1 - 52.2 MB)
├── best_model_with_attn.pt  # Checkpoint trọng số tối ưu (Mô hình 2 - 75.4 MB)
├── data_50k_clean.tsv       # Tập dữ liệu huấn luyện tiền xử lý (50.000 cặp câu Anh-Việt)
├── test_set_2500.tsv        # Tập dữ liệu kiểm thử độc lập (2.500 câu)
├── train.ipynb              # Toàn bộ quy trình tiền xử lý, huấn luyện và đánh giá BLEU
└── train_full_cuda.ipynb    # Bản notebook tối ưu hóa tăng tốc trên GPU NVIDIA CUDA
```

---

## 🚀 Hướng dẫn Chạy Thử nghiệm (Inference)

### 1. Cài đặt Thư viện
```bash
pip install torch torchvision torchaudio nltk tqdm
```

### 2. Dịch thử nghiệm câu tiếng Anh (Greedy Search)
```python
import torch

# Nạp mô hình có Attention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_with_attn = ... # Khởi tạo kiến trúc Seq2SeqAttention
model_with_attn.load_state_dict(torch.load("best_model_with_attn.pt", map_location=device))
model_with_attn.eval()

# Dịch câu mẫu
english_sentence = "Artificial intelligence is transforming modern software engineering."
vietnamese_translation = translate_with_attn(english_sentence, model_with_attn, vocab_src, vocab_tgt)
print(f"Bản dịch tiếng Việt: {vietnamese_translation}")
```

---

## 📊 Đánh giá Thực nghiệm
- Sử dụng độ đo **BLEU Score** (Bilingual Evaluation Understudy) trên tập kiểm thử 2.500 câu độc lập.
- Cơ chế Bahdanau Attention giúp giảm Loss nhanh hơn, hội tụ tốt hơn và điểm BLEU cao hơn rõ rệt so với mô hình thuần Encoder-Decoder.

---

## 👤 Tác giả
- **Tác giả**: Cao Đức Trung (`namnguyenit`)
- **Nền tảng**: PyTorch CUDA Deep Learning Framework
