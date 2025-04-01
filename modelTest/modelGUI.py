import os
import re
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from time import time

from PIL import Image, ImageTk


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # ========== 窗口配置 ==========
        self.title("可重构工具链执行界面")
        self.geometry("1100x750")
        self.minsize(900, 600)
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
        title_label.pack(pady=(0, 20))

        # ========== 控制栏容器 ==========
        control_bar = ttk.Frame(main_frame)
        control_bar.pack(pady=10, fill=tk.X, anchor="nw")  # 使用fill=X确保横向填充

        # ========== 图片选择框 ==========
        self.image_path = ""
        self.canvas = tk.Canvas(
            control_bar,  # 父容器改为control_bar
            width=300,
            height=200,
            bg="white",
            highlightthickness=1,
            highlightbackground="#ccc"
        )

        self.canvas.pack(side=tk.LEFT, padx=(10,80))  # 左对齐并添加间距
        self.canvas.bind("<Button-1>", self.select_image)
        self.update_idletasks()
        self.draw_crosshair()

        # ========== 控制按钮区 ==========
        button_frame = ttk.Frame(control_bar)
        button_frame.pack(side=tk.LEFT, padx=80, anchor="center")  # 左对齐并添加间距

        # 优化选项
        self.optimize_var = tk.BooleanVar()
        optimize_check = ttk.Checkbutton(
            button_frame,  # 父容器改为button_frame
            text="优化",
            variable=self.optimize_var,
            style="Custom.TCheckbutton"
        )
        optimize_check.pack(pady=5)  # 垂直排列

        # 运行按钮
        run_btn = tk.Button(
            button_frame,  # 父容器改为button_frame
            text="执行",
            command=self.run_code,
            bg="#007bff",
            fg="black",
            font=("Arial", 14, "bold"),  # 调小字体
            activebackground="#0056b3",
            relief="flat",
            padx=15,
            pady=5,
            width=12  # 调整宽度
        )
        run_btn.pack(pady=5)
        # run_btn.pack(side=tk.LEFT, padx=20)

        # ========== 输出区域 ==========
        output_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        output_paned.pack(fill=tk.BOTH, expand=True)

        # 左侧 - 执行日志
        left_frame = ttk.Frame(output_paned,width=350)
        left_frame.pack_propagate(False)  # 禁止子控件改变框架大小
        log_label = ttk.Label(left_frame, text="执行日志:", font=("Arial", 12))
        log_label.pack(anchor="w")

        self.log_text = tk.Text(
            left_frame,
            wrap=tk.WORD,
            bg="white",
            font=('Consolas', 11),
            padx=10,
            pady=10,
            height=15
        )
        log_scroll = ttk.Scrollbar(left_frame, command=self.log_text.yview)

        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        

        # 右侧 - 结果记录
        right_frame = ttk.Frame(output_paned)
        result_label = ttk.Label(right_frame, text="执行结果:", font=("Arial", 12))
        result_label.pack(anchor="w")

        # 结果表格
        self.result_table = ttk.Treeview(
            right_frame,
            columns=("选择", "ID", "图片路径", "输出概率", "是否优化", "输出时间"),
            show="headings",
            height=15,
            selectmode="none"
        )

        # 设置列

        columns = {
            "选择": {"width": 40, "anchor": "center"},  # 空标题
            "ID": {"width": 40, "anchor": "center"},
            "图片路径": {"width": 80},
            "输出概率": {"width": 200},
            "是否优化": {"width": 60, "anchor": "center"},
            "输出时间": {"width": 60, "anchor": "center"}
        }

        for col, config in columns.items():
            self.result_table.heading(col, text=col)
            self.result_table.column(col, **config)

        # 添加复选框
        self.result_table.tag_configure("selected", background="#e6f3ff")
        self.checkboxes = {}  # 存储复选框变量

        # 滚动条
        result_scroll = ttk.Scrollbar(right_frame, command=self.result_table.yview)
        self.result_table.configure(yscrollcommand=result_scroll.set)
        self.result_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 优化比计算区域
        calc_frame = ttk.Frame(right_frame)
        calc_frame.pack(fill=tk.X, pady=(10, 0))

        calc_btn = tk.Button(
            calc_frame,
            text="优化比计算",
            command=self.calculate_ratio,
            bg="#007bff",
            fg="black",
            font=("Arial", 14, "bold"),  # 调小字体
            activebackground="#0056b3"
        )
        calc_btn.pack(pady=5)

        self.ratio_label = ttk.Label(
            calc_frame,
            text="等待计算...",
            font=("Arial", 12)
        )
        self.ratio_label.pack()

        output_paned.add(left_frame,weight=0)
        output_paned.add(right_frame,weight=1)

        # 自定义样式
        self.style = ttk.Style()
        self.style.configure("Custom.TCheckbutton", font=("Arial", 12))
        self.style.configure("Accent.TButton", font=("Arial", 12), foreground="white", background="#007bff")
        self.style.map("Accent.TButton",
                       foreground=[('pressed', 'white'), ('active', 'white')],
                       background=[('pressed', '#0056b3'), ('active', '#0069d9')])

        # 存储执行记录
        self.execution_records = []
        self.record_id = 1

    # def select_image(self):
    #     """选择图片文件"""
    #     filetypes = [("图片文件", "*.jpg *.jpeg *.png *.bmp")]
    #     path = filedialog.askopenfilename(title="选择图片", filetypes=filetypes)
    #     if path:
    #         self.image_path_entry.delete(0, tk.END)
    #         self.image_path_entry.insert(0, path)
    #     self.replace_image_path_in_tvm(self.image_path_entry.get())
    def select_image(self, event=None):
        """选择图片文件"""
        filetypes = [("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        path = filedialog.askopenfilename(title="选择图片", filetypes=filetypes)

        if path:
            self.image_path = path
            try:
                # 显示缩略图
                img = Image.open(path)
                img.thumbnail((300, 200))
                photo = ImageTk.PhotoImage(img)

                # 清除画布并显示图片
                self.canvas.delete("all")
                self.canvas.create_image(
                    self.canvas.winfo_width() / 2,
                    self.canvas.winfo_height() / 2,
                    image=photo,
                    anchor="center"
                )
                self.canvas.image = photo  # 保持引用

                # 更新tvm文件中的路径
                self.replace_image_path_in_tvm(path)

            except Exception as e:
                messagebox.showerror("错误", f"无法加载图片: {str(e)}")
                self.draw_crosshair()

    def draw_crosshair(self):
        """绘制十字线"""
        # w = self.canvas.winfo_width()
        # h = self.canvas.winfo_height()

        #使用硬编码
        w,h=300,200

        print(f"w={w},h={h}")

        # 清除之前的内容
        self.canvas.delete("all")

        # 绘制十字线
        self.canvas.create_line(w / 2, 0, w / 2, h, fill="#e0e0e0", dash=(4, 2))
        self.canvas.create_line(0, h / 2, w, h / 2, fill="#e0e0e0", dash=(4, 2))

        # 添加提示文字
        self.canvas.create_text(
            w / 2, h / 2,
            text="点击选择图片",
            fill="#a0a0a0",
            font=("Arial", 12)
        )

    def replace_image_path_in_tvm(self, new_path):
        """替换tvm11.py文件中的图片路径"""
        #当前文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tvm_path = os.path.join(current_dir, "tvm11.py")
        print(tvm_path)
        if not os.path.exists(tvm_path):
            messagebox.showerror("错误", "tvm11.py 文件不存在")
            return
        try:
            with open(tvm_path, 'r', encoding='utf-8') as file:
                code = file.read()


            # 使用正则表达式找到并替换图片路径
            code = re.sub(r'image=Image\.open\(["\'](.*?)["\']\)', f'image=Image.open("{new_path}")', code)
            print(code)
            with open(tvm_path, 'w', encoding='utf-8') as file:
                file.write(code)

        except Exception as e:
            self.log_text.insert(tk.END, f"替换图片路径时出错: {str(e)}\n")
            messagebox.showerror("错误", f"替换图片路径时发生错误: {str(e)}")

    def run_code(self):
        """执行代码"""
        image_path = self.image_path
        if not image_path:
            messagebox.showwarning("警告", "请先选择图片路径")
            return

        optimize = self.optimize_var.get()
        start_time = time()
        probability = "[]"  # 默认值

        try:
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"开始执行 {'优化' if optimize else '普通'} 配置...\n")
            self.log_text.insert(tk.END, f"图片路径: {image_path}\n")
            self.update()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            commands = [
                f"python 2.py tvm11.py output_model_info.json 1.bin {int(optimize)}",
                "python HNU-W-DDR_dma.py 1.bin",
                "./1.sh",
                "./2.sh",
                "python HNU-R-DDR.py 1.bin 1.txt",
            ]

            for cmd in commands:
                self.log_text.insert(tk.END, f">> 执行：{cmd}\n")
                self.update()

                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    cwd=current_dir
                )
                self.log_text.insert(tk.END, result.stdout + "\n")
                self.update()

                # 解析 HNU-R-DDR.py 的输出
                if "HNU-R-DDR.py" in cmd:
                    probability = self.parse_probability_output(result.stdout)

            # 计算执行时间
            exec_time = round(time() - start_time, 2)

            # 添加到结果表格
            item_id = self.result_table.insert("", "end",
                                               values=("",  # 复选框占位
                                                       self.record_id,
                                                       os.path.basename(image_path),
                                                       probability,
                                                       "是" if optimize else "否",
                                                       f"{exec_time:.2f}"),
                                               tags=("record",))

            # 添加实际复选框
            var = tk.BooleanVar()
            self.checkboxes[item_id] = var
            self.result_table.set(item_id, "选择", "")
            self.create_checkbox(item_id)

            # 保存记录
            self.execution_records.append({
                "id": self.record_id,
                "image_path": image_path,
                "probability": probability,
                "optimized": optimize,
                "exec_time": exec_time
            })
            self.record_id += 1

            self.log_text.insert(tk.END, f"\n执行完成！耗时：{exec_time:.2f}s\n")
            messagebox.showinfo("完成", "执行成功完成")

        except Exception as e:
            self.log_text.insert(tk.END, f"\n执行出错: {str(e)}\n")
            messagebox.showerror("错误", f"执行过程中发生错误: {str(e)}")

    def create_checkbox(self, item_id):
        """为表格行创建复选框（修复版）"""
        var = tk.BooleanVar()
        self.checkboxes[item_id] = var  # 确保变量被存储

        # 使用 tk.Checkbutton 而不是 ttk.Checkbutton
        cb = tk.Checkbutton(
            self.result_table,
            variable=var,
            bg='white',  # 背景色与表格一致
            relief='flat',  # 扁平样式
            command=lambda: self.on_checkbox_click(item_id)
        )# 点击回调
        # 定位复选框
        bbox = self.result_table.bbox(item_id, "选择")
        if bbox:
            x, y, w, h = bbox
        cb.place(in_=self.result_table, x=x, y=y, width=w, height=h)

        # 初始设置值
        self.result_table.set(item_id, "选择", "☑" if var.get() else "☐")

    def on_checkbox_click(self, item_id):
        """复选框点击回调"""
        var = self.checkboxes[item_id]
        # 更新显示符号
        self.result_table.set(item_id, "选择", "☑" if var.get() else "☐")

        # 限制只能选择两个
        selected = [k for k, v in self.checkboxes.items() if v.get()]
        if len(selected) > 2:
            var.set(False)  # 取消当前选择
            self.result_table.set(item_id, "选择", "☐")
            messagebox.showwarning("提示", "最多只能选择两条记录进行比较")

    def parse_probability_output(self, output):
        """解析概率输出，返回格式化后的概率数组字符串"""
        try:
            # 使用正则表达式提取方括号内的内容
            match = re.search(r'\[([^\]]+)\]', output)
            if not match:
                return "解析失败"

            prob_str = match.group(1)
            try:
                probabilities = [float(x.strip()) for x in prob_str.split(',')]
                # 格式化数组为字符串，保留2位小数
                return "[" + ", ".join([f"{x:.2f}" for x in probabilities]) + "]"
            except ValueError as e:
                return f"数值错误: {str(e)}"
        except Exception as e:
            return f"解析异常: {str(e)}"

    def calculate_ratio(self):
        """计算优化比（修复版）"""
        selected = [k for k, v in self.checkboxes.items() if v.get()]

        if len(selected) != 2:
            messagebox.showwarning("警告", "请选择两条记录进行比较")
            return

        # 获取记录数据
        records = []
        for item_id in selected:
            values = self.result_table.item(item_id)['values']
            records.append({
                "time": float(values[5]),  # 输出时间
                "optimized": values[4] == "是"  # 是否优化
            })

        time1, time2 = records[0]["time"], records[1]["time"]
        ratio = round((max(time1, time2)-min(time1,time2)) / max(time1, time2), 2)

        # 判断哪个更快
        # if time1 < time2:
        #     faster = "配置1" if records[0]["optimized"] else "配置2"
        # else:
        #     faster = "配置2" if records[1]["optimized"] else "配置1"

        result_text = f"优化比: {ratio*100}%"
        self.ratio_label.config(text=result_text)
        self.log_text.insert(tk.END, f"\n优化比计算结果: {result_text}\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()