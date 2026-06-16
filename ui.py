import json
import os
import re
import tkinter as tk
from tkinter import ttk, messagebox

import torch

from model_seq2seq import Seq2SeqAttention
from model_transformer import TransformerCalc
from vocab import VOCAB_SIZE, tokenize_input, decode_output

INPUT_RE = re.compile(r'^(\d+)([+\-])(\d+)$')

# ── colour palette ──────────────────────────────────────────────────
CLR_DISPLAY_BG  = "#1a1a2e"
CLR_DISPLAY_FG  = "#e0e0e0"
CLR_RESULT_BG   = "#16213e"
CLR_BTN_DIGIT   = ("#f0f0f0", "#222")
CLR_BTN_OP      = ("#4a90e2", "#fff")
CLR_BTN_ACTION  = ("#e2914a", "#fff")
CLR_OK          = "#2e7d32"
CLR_ERR         = "#c62828"
CLR_INFO        = "#888888"


class AICalcUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AICalc — Neural Calculator")
        self.root.minsize(700, 440)
        self.root.resizable(True, True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_type_loaded = None
        self.current_expr = ""
        self.just_calculated = False
        self.total = 0
        self.correct = 0

        self._build_ui()
        self._bind_keys()
        self._refresh_checkpoints()
        self.root.mainloop()

    # ── layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        f = ttk.LabelFrame(self.root, text=" Model ", padding=10)
        f.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        # model selector
        self.model_var = tk.StringVar(value="seq2seq")
        ttk.Radiobutton(f, text="Seq2Seq + Attention",
                        variable=self.model_var, value="seq2seq",
                        command=self._refresh_checkpoints).pack(anchor="w", pady=2)
        ttk.Radiobutton(f, text="Transformer",
                        variable=self.model_var, value="transformer",
                        command=self._refresh_checkpoints).pack(anchor="w", pady=2)

        ttk.Separator(f).pack(fill="x", pady=8)

        # checkpoint selector
        ttk.Label(f, text="Checkpoint:").pack(anchor="w")
        self.ckpt_var = tk.StringVar()
        self.ckpt_combo = ttk.Combobox(f, textvariable=self.ckpt_var,
                                        state="readonly", width=24)
        self.ckpt_combo.pack(fill="x", pady=(2, 6))
        ttk.Button(f, text="Load Model", command=self._load_model).pack(fill="x")

        self.status_lbl = ttk.Label(f, text="No model loaded",
                                     foreground="red", wraplength=175,
                                     justify="left")
        self.status_lbl.pack(anchor="w", pady=(6, 0))

        ttk.Separator(f).pack(fill="x", pady=8)

        # stats
        ttk.Label(f, text="Session Stats",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        self.stats_var = tk.StringVar(value="No calculations yet.")
        ttk.Label(f, textvariable=self.stats_var, justify="left").pack(anchor="w", pady=(2, 6))
        ttk.Button(f, text="Reset Stats", command=self._reset_stats).pack(fill="x")

    def _build_center_panel(self):
        f = ttk.LabelFrame(self.root, text=" Calculator ", padding=10)
        f.grid(row=0, column=1, sticky="n", padx=5, pady=10)

        # ── display ────────────────────────────────────────────────
        df = tk.Frame(f, bg=CLR_DISPLAY_BG, bd=2, relief="sunken")
        df.pack(fill="x", pady=(0, 10))

        self.expr_var = tk.StringVar()
        tk.Entry(df, textvariable=self.expr_var,
                 font=("Courier", 20), justify="right",
                 state="readonly", bd=0,
                 bg=CLR_DISPLAY_BG, fg=CLR_DISPLAY_FG,
                 readonlybackground=CLR_DISPLAY_BG,
                 width=13).pack(fill="x", padx=6, pady=(6, 2))

        self.result_var = tk.StringVar(value="")
        self.result_lbl = tk.Label(df, textvariable=self.result_var,
                                    font=("Courier", 13), anchor="e",
                                    bg=CLR_RESULT_BG, fg=CLR_INFO)
        self.result_lbl.pack(fill="x", padx=6, pady=(0, 6))

        # ── number pad ─────────────────────────────────────────────
        gf = tk.Frame(f)
        gf.pack()

        pad = [
            [("7","d"), ("8","d"), ("9","d"), ("+","o")],
            [("4","d"), ("5","d"), ("6","d"), ("-","o")],
            [("1","d"), ("2","d"), ("3","d"), ("=","a")],
            [("0","d"), ("⌫","a"), ("C","a"),  None     ],
        ]
        clr = {"d": CLR_BTN_DIGIT, "o": CLR_BTN_OP, "a": CLR_BTN_ACTION}

        for r, row in enumerate(pad):
            for c, cell in enumerate(row):
                if cell is None:
                    continue
                txt, kind = cell
                bg, fg = clr[kind]
                btn = tk.Button(gf, text=txt, width=4, height=2,
                                font=("TkDefaultFont", 14, "bold"),
                                bg=bg, fg=fg, activebackground=bg,
                                relief="flat", bd=0,
                                command=lambda t=txt: self._press(t))
                btn.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")

    def _build_right_panel(self):
        f = ttk.LabelFrame(self.root, text=" History ", padding=10)
        f.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)

        self.history = tk.Text(f, font=("Courier", 10), state="disabled",
                                wrap="none", bg="#fafafa", relief="flat")
        sb = ttk.Scrollbar(f, command=self.history.yview)
        self.history.configure(yscrollcommand=sb.set)
        self.history.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self.history.tag_config("correct", foreground=CLR_OK)
        self.history.tag_config("wrong",   foreground=CLR_ERR)
        self.history.tag_config("info",    foreground=CLR_INFO)

    # ── keyboard ────────────────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("<KeyPress>", self._on_key)

    def _on_key(self, event):
        # ignore keypresses that target a real input widget
        focused = self.root.focus_get()
        if isinstance(focused, (ttk.Combobox,)):
            return

        k, ch = event.keysym, event.char
        if ch in "0123456789":
            self._press(ch)
        elif ch in "+-":
            self._press(ch)
        elif k in ("Return", "KP_Enter") or ch == "=":
            self._press("=")
        elif k == "BackSpace":
            self._press("⌫")
        elif k == "Escape":
            self._press("C")

    # ── button logic ────────────────────────────────────────────────

    def _press(self, text):
        if text == "C":
            self.current_expr = ""
            self.just_calculated = False
            self.expr_var.set("")
            self.result_var.set("")
            self.result_lbl.config(fg=CLR_INFO)

        elif text == "⌫":
            # backspace after a calculation: keep the expression, just let the user edit
            if self.just_calculated:
                self.just_calculated = False
                self.result_var.set("")
                self.result_lbl.config(fg=CLR_INFO)
            self.current_expr = self.current_expr[:-1]
            self.expr_var.set(self.current_expr)

        elif text in "+-":
            # any operator press after a calculation starts fresh
            if self.just_calculated:
                self.current_expr = ""
                self.just_calculated = False
                self.result_var.set("")
                self.result_lbl.config(fg=CLR_INFO)
            if not self.current_expr:
                return
            if "+" in self.current_expr or "-" in self.current_expr:
                return
            self.current_expr += text
            self.expr_var.set(self.current_expr)

        elif text == "=":
            self._calculate()

        else:  # digit — wipe previous result and start fresh
            if self.just_calculated:
                self.current_expr = ""
                self.just_calculated = False
                self.result_var.set("")
                self.result_lbl.config(fg=CLR_INFO)
            self.current_expr += text
            self.expr_var.set(self.current_expr)

    # ── inference ───────────────────────────────────────────────────

    def _calculate(self):
        expr = self.current_expr.strip()
        if not expr:
            return

        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return

        m = INPUT_RE.match(expr)
        if not m:
            self._set_result("Need:  number  +/-  number", CLR_ERR)
            return

        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if not (0 <= a <= 127 and 0 <= b <= 127):
            self._set_result("Both numbers must be 0 – 127", CLR_ERR)
            return

        gt = str((a + b) % 128 if op == "+" else (a - b) % 128)

        toks = torch.tensor([tokenize_input(expr)], dtype=torch.long, device=self.device)
        if self.model_type_loaded == "seq2seq":
            lens = torch.tensor([toks.size(1)])
            pred = decode_output(self.model.inference(toks, lens, max_len=10)[0])
        else:
            pred = decode_output(self.model.inference(toks, max_len=10)[0])

        ok = (pred == gt)
        self.total += 1
        if ok:
            self.correct += 1

        mark = "✓" if ok else "✗"
        self._set_result(f"= {pred}    GT: {gt}   {mark}", CLR_OK if ok else CLR_ERR)
        self.just_calculated = True

        pct = 100 * self.correct / self.total
        self.stats_var.set(f"{self.correct} / {self.total} correct\n{pct:.1f}% accuracy")

        self._log(f"{expr} = {pred}   (GT: {gt})  {mark}", "correct" if ok else "wrong")

    def _set_result(self, text, color):
        self.result_var.set(text)
        self.result_lbl.config(fg=color)

    # ── model loading ───────────────────────────────────────────────

    def _refresh_checkpoints(self):
        folder = os.path.join("models", self.model_var.get())
        ckpts = []
        if os.path.isdir(folder):
            ckpts = sorted(
                f for f in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, f))
            )
        self.ckpt_combo["values"] = ckpts
        if ckpts:
            self.ckpt_combo.current(len(ckpts) - 1)   # default to most recent
        else:
            self.ckpt_var.set("")

    def _load_model(self):
        mtype = self.model_var.get()
        ckpt  = self.ckpt_var.get()
        if not ckpt:
            messagebox.showerror("Error", "No checkpoint selected.")
            return

        folder = os.path.join("models", mtype, ckpt)
        try:
            with open(os.path.join(folder, "config.json")) as fh:
                cfg = json.load(fh)

            if mtype == "seq2seq":
                model = Seq2SeqAttention(
                    vocab_size=VOCAB_SIZE,
                    embed_size=int(cfg["embed_size"]),
                    hidden_size=int(cfg["hidden_size"]),
                    num_layers=int(cfg["num_layers"]),
                ).to(self.device)
            else:
                model = TransformerCalc(
                    vocab_size=VOCAB_SIZE,
                    d_model=int(cfg["d_model"]),
                    nhead=int(cfg["nhead"]),
                    num_encoder_layers=int(cfg["num_encoder_layers"]),
                    num_decoder_layers=int(cfg["num_decoder_layers"]),
                    dim_feedforward=int(cfg["dim_feedforward"]),
                    dropout=float(cfg["dropout"]),
                ).to(self.device)

            model.load_state_dict(
                torch.load(os.path.join(folder, "model.pt"), map_location=self.device)
            )
            model.eval()
            self.model = model
            self.model_type_loaded = mtype
            self.status_lbl.config(text=f"✓  {mtype}\n{ckpt}", foreground="green")
            self._log(f"Loaded {mtype}: {ckpt}", "info")

        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    # ── history / stats ─────────────────────────────────────────────

    def _log(self, text, tag="info"):
        self.history.config(state="normal")
        self.history.insert("end", text + "\n", tag)
        self.history.see("end")
        self.history.config(state="disabled")

    def _reset_stats(self):
        self.total = 0
        self.correct = 0
        self.stats_var.set("No calculations yet.")
        self._log("─── stats reset ───", "info")


if __name__ == "__main__":
    AICalcUI()
