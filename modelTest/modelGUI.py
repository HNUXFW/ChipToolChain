import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # ========== 窗口配置 ==========
        self.title("可重构芯片")
        self.geometry("800x600")
        self.minsize(700, 500)
        self.configure(bg="#f0f0f0")

        # ========== 主框架 ==========
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # ========== 标题 ==========
        title_label = ttk.Label(
            main_frame,
            text="可重构工具链执行界面",
            font=("Arial", 20, "bold"),
            foreground="#2c3e50",
            anchor="center"
        )
        title_label.pack(pady=(0, 30))  # 减少标题下方间距

        # ========== 控制区 ==========
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10, fill=tk.X)
        self.style = ttk.Style()
        # 1. 增大选择框(indicator)尺寸
        self.style.configure(
            "TCheckbutton",
            indicatorsize=20  # 调整选择框大小（默认约12px）
        )

        # 2. 增大字体和间距
        self.style.configure(
            "TCheckbutton",
            font=("Arial", 16),  # 原字体可能为10pt
            padding=(5, 5)  # 增加内边距
        )
        # 优化复选框（上方）
        self.optimize_var = tk.IntVar(value=0)
        optimize_checkbox = ttk.Checkbutton(
            control_frame,
            text="优化",
            variable=self.optimize_var,
            style="TCheckbutton"
        )
        optimize_checkbox.pack(pady=(0, 15))  # 增加与按钮的间距

        # 运行按钮（下方）
        run_button = tk.Button(
            control_frame,
            text="运行",
            command=self.run_code,
            bg="#007bff",  # 蓝色背景
            fg="black",  # 黑色字体
            font=("Arial", 18, "bold"),
            activebackground="#0056b3",
            relief="flat",
            padx=20,
            pady=10,
            width=18
        )
        run_button.pack()

        # ========== 输出结果区域 ==========
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        self.result_text = tk.Text(
            output_frame,
            wrap=tk.WORD,
            bg="white",
            font=('Consolas', 12),
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        scrollbar = ttk.Scrollbar(output_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def run_code(self):
        optimize = self.optimize_var.get()
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)

            commands = [
                f"python 2.py tvm11.py output_model_info.json 1.bin {optimize}",
                "python HNU-W-DDR_dma.py 1.bin",
                "./1.sh",
                "./2.sh",
                "python HNU-R-DDR.py 1.bin 1.txt",

            ]

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END)

            for cmd in commands:
                self.result_text.insert(tk.END, f">>> 正在执行: {cmd}\n", "command")
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8"
                )
                self.result_text.insert(tk.END, result.stdout + "\n", "output")
                self.result_text.see(tk.END)
                self.update_idletasks()

            self.result_text.config(state=tk.DISABLED)

        except subprocess.CalledProcessError as e:
            messagebox.showerror("错误", f"命令执行失败: {e.cmd}\n错误信息: {e.stderr}")
        except Exception as e:
            messagebox.showerror("错误", f"发生未知错误: {str(e)}")


if __name__ == "__main__":
    app = App()
    app.mainloop()