import nbformat as nbf

notebook_path = r"D:\Downloads\trainsq2sq\train.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

# ----------------- CELL 25: Markdown - Thử nghiệm dịch thực tế -----------------
c25_md = """## 12. Thử Nghiệm Dịch Thực Tế (Inference / Translation Demo)
Hàm dịch câu tiếng Anh sang tiếng Việt sử dụng thuật toán **Greedy Search (sinh từ tuần tự)**.
Chúng ta sẽ nạp trọng số tốt nhất (`best_model_no_attn.pt` và `best_model_with_attn.pt`) để đối chiếu kết quả dịch của cả 2 mô hình.
"""

# ----------------- CELL 26: Code - Thử nghiệm dịch thực tế -----------------
c26_code = """# =====================================================================
# HÀM SUY LUẬN & DỊCH THỬ NGHIỆM CÂU TIẾNG ANH (INFERENCE)
# =====================================================================

# 1. Nạp lại trọng số tốt nhất của 2 mô hình
model_no_attn.load_state_dict(torch.load("best_model_no_attn.pt", map_location=device))
model_with_attn.load_state_dict(torch.load("best_model_with_attn.pt", map_location=device))

model_no_attn.eval()
model_with_attn.eval()

# 2. Hàm dịch cho Mô hình 1 (Không Attention)
def translate_no_attn(sentence, model, vocab_src, vocab_tgt, max_len=30):
    model.eval()
    tokens = vocab_src.tokenize(sentence)
    indices = [vocab_src.SOS_IDX] + [vocab_src.word2idx.get(t, vocab_src.UNK_IDX) for t in tokens] + [vocab_src.EOS_IDX]
    src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indices = [vocab_tgt.SOS_IDX]
    for _ in range(max_len):
        prev_token = torch.tensor([trg_indices[-1]], dtype=torch.long).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(prev_token, hidden, cell)
        
        pred_token = output.argmax(1).item()
        if pred_token == vocab_tgt.EOS_IDX:
            break
        trg_indices.append(pred_token)

    return vocab_tgt.decode(trg_indices)

# 3. Hàm dịch cho Mô hình 2 (Có Attention)
def translate_with_attn(sentence, model, vocab_src, vocab_tgt, max_len=30):
    model.eval()
    tokens = vocab_src.tokenize(sentence)
    indices = [vocab_src.SOS_IDX] + [vocab_src.word2idx.get(t, vocab_src.UNK_IDX) for t in tokens] + [vocab_src.EOS_IDX]
    src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_states, decoder_hidden = model.encoder(src_tensor)

    trg_indices = [vocab_tgt.SOS_IDX]
    for _ in range(max_len):
        prev_token = torch.tensor([trg_indices[-1]], dtype=torch.long).to(device)
        with torch.no_grad():
            output, decoder_hidden = model.decoder(prev_token, decoder_hidden, encoder_states)

        pred_token = output.argmax(1).item()
        if pred_token == vocab_tgt.EOS_IDX:
            break
        trg_indices.append(pred_token)

    return vocab_tgt.decode(trg_indices)

# 4. Danh sách các câu mẫu để kiểm tra
test_sentences = [
    "What are you talking about?",
    "Who are you crazy for?",
    "Why is he so big?",
    "Please stand behind the yellow line.",
    "How are you today?",
    "You will need to view:",
    "Support quickly."
]

print("=" * 80)
print(f"{'CÂU TIẾNG ANH (EN)':<35} | {'MÔ HÌNH KHÔNG ATTENTION':<20} | {'MÔ HÌNH CÓ ATTENTION'}")
print("=" * 80)

for sent in test_sentences:
    trans_1 = translate_no_attn(sent, model_no_attn, vocab_en, vocab_vi)
    trans_2 = translate_with_attn(sent, model_with_attn, vocab_en, vocab_vi)
    print(f"{sent:<35} | {trans_1:<20} | {trans_2}")

print("=" * 80)
"""

new_cells = [
    nbf.v4.new_markdown_cell(c25_md),
    nbf.v4.new_code_cell(c26_code)
]

nb.cells.extend(new_cells)

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Added translation demo cells! Total cells: {len(nb.cells)}")

