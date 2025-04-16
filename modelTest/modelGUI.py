import os
import re
import subprocess
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.constants import HORIZONTAL

import ttkbootstrap as ttk
from time import time

from PIL import Image, ImageTk


class App(ttk.Window):
    def __init__(self):
        super().__init__(themename="minty")

        # ========== 窗口配置 ==========
        self.title("可重构工具链执行界面")
        self.geometry("1600x900")
        self.minsize(900, 600)
        self.configure(bg="#f0f0f0")

        # ========== 主框架 ==========
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # ========== 标题 ==========
        title_label = ttk.Label(
            main_frame,
            text="可重构工具链执行界面",
            font=("TkDefaultFont", 20, "bold"),
            foreground="#2c3e50",
            anchor="center"
        )
        title_label.pack(pady=(0, 20))

        # ========== 主内容区域 ==========
        content_frame = ttk.PanedWindow(main_frame, orient=HORIZONTAL)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # ========== 左侧区域 ==========
        left_frame = ttk.Frame(content_frame)
        content_frame.add(left_frame, weight=1)

        # 左侧上方 - 图片选择和执行按钮
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(0, 20))

        # 图片选择框
        self.image_path = ""
        canvas_frame = ttk.Frame(control_frame, borderwidth=2, relief="solid")
        canvas_frame.pack(side=tk.LEFT)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=300,
            height=200,
            bg="#f0f0f0",
            highlightthickness=2,
            highlightbackground="#ccc"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.select_image)
        self.draw_crosshair()

        # 控制按钮区
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=50, pady=50, anchor="n")

        # 优化选项
        self.optimize_var = tk.BooleanVar()
        optimize_check = ttk.Checkbutton(
            button_frame,
            text="优化",
            variable=self.optimize_var,
            style="Custom.TCheckbutton"
        )
        optimize_check.pack(pady=5, anchor=tk.CENTER)

        # 运行按钮
        run_btn = tk.Button(
            button_frame,
            text="执行",
            command=self.run_code,
            bg="#007bff",
            fg="black",
            font=("TkDefaultFont", 14, "bold"),
            activebackground="#0056b3",
            relief="flat",
            padx=15,
            pady=5,
            width=12
        )
        run_btn.pack(pady=10, anchor=tk.CENTER)

        config_btn = tk.Button(
            button_frame,
            text="配置",
            command=self.run_code,
            bg="#007bff",
            fg="black",
            font=("TkDefaultFont", 14, "bold"),
            activebackground="#0056b3",
            relief="flat",
            padx=15,
            pady=5,
            width=12
        )
        config_btn.pack(pady=10, anchor=tk.CENTER)

        # 左侧下方 - 执行日志
        log_frame = ttk.Frame(left_frame)
        log_frame.pack_propagate(False)
        log_frame.pack(fill=tk.BOTH, expand=True)

        log_label = ttk.Label(
            log_frame,
            text="执行日志:",
            font=("YaHei", 12)
        )
        log_label.pack(anchor="w", padx=5, pady=(0, 5))

        self.log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            bg="white",
            font=('Consolas', 11),
            padx=10,
            pady=10
        )
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ========== 右侧区域 - 执行结果 ==========
        right_frame = ttk.Frame(content_frame)
        content_frame.add(right_frame, weight=5)

        # 创建分割左右结果的PanedWindows
        result_paned = ttk.PanedWindow(right_frame, orient=tk.HORIZONTAL)
        result_paned.pack(fill=tk.BOTH, expand=True)

        # ========== 左侧结果 - 未优化 ==========
        unoptimized_frame = ttk.Frame(result_paned)
        result_paned.add(unoptimized_frame, weight=1)

        unoptimized_label = ttk.Label(
            unoptimized_frame,
            text="未优化结果:",
            font=("YaHei", 12),
            anchor="w"
        )
        unoptimized_label.pack(fill=tk.X, padx=5, pady=(0, 5))

        unoptimized_table_frame = ttk.Frame(unoptimized_frame)
        unoptimized_table_frame.pack(fill=tk.BOTH, expand=True)

        self.unoptimized_table = ttk.Treeview(
            unoptimized_table_frame,
            columns=("选择", "ID", "输出概率", "输出时间"),
            show="headings",
            selectmode="none"
        )
        self.unoptimized_table.tag_configure('checked', foreground='black')
        self.unoptimized_table.tag_configure('unchecked', foreground='black')

        unoptimized_scroll = ttk.Scrollbar(unoptimized_table_frame, command=self.unoptimized_table.yview)
        self.unoptimized_table.configure(yscrollcommand=unoptimized_scroll.set)
        self.unoptimized_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        unoptimized_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 设置未优化表格列
        unoptimized_columns = {
            "选择": {"width": 50, "stretch": False},
            "ID": {"width": 50, "stretch": False},
            "输出概率": {"width": 180, "stretch": True, "anchor": "center"},
            "输出时间": {"width": 90, "stretch": False}
        }

        for col, config in unoptimized_columns.items():
            self.unoptimized_table.heading(col, text=col)
            self.unoptimized_table.column(col, **config)

        # ========== 右侧结果 - 优化后 ==========
        optimized_frame = ttk.Frame(result_paned)
        result_paned.add(optimized_frame, weight=1)

        optimized_label = ttk.Label(
            optimized_frame,
            text="优化后结果:",
            font=("YaHei", 12),
            anchor="w"
        )
        optimized_label.pack(fill=tk.X, padx=5, pady=(0, 5))

        optimized_table_frame = ttk.Frame(optimized_frame)
        optimized_table_frame.pack(fill=tk.BOTH, expand=True)

        self.optimized_table = ttk.Treeview(
            optimized_table_frame,
            columns=("选择", "ID", "输出概率", "输出时间"),
            show="headings",
            selectmode="none"
        )
        self.optimized_table.tag_configure('checked', foreground='black')
        self.optimized_table.tag_configure('unchecked', foreground='black')

        optimized_scroll = ttk.Scrollbar(optimized_table_frame, command=self.optimized_table.yview)
        self.optimized_table.configure(yscrollcommand=optimized_scroll.set)
        self.optimized_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        optimized_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 设置优化表格列
        optimized_columns = {
            "选择": {"width": 50, "stretch": False},
            "ID": {"width": 50, "stretch": False},
            "输出概率": {"width": 180, "stretch": True, "anchor": "center"},
            "输出时间": {"width": 90, "stretch": False}
        }
        self.optimized_table.column("#0", width=0, stretch=tk.NO)

        for col, config in optimized_columns.items():
            self.optimized_table.heading(col, text=col)
            self.optimized_table.column(col, **config)

        # 绑定点击事件
        self.unoptimized_table.tag_bind('record', '<Button-1>', self.on_cell_click)
        self.optimized_table.tag_bind('record', '<Button-1>', self.on_cell_click)

        # 存储执行记录
        self.execution_records = []
        self.record_id = 1
        self.selection_state = {'optimized': {}, 'unoptimized': {}}

        # ========== 底部 - 优化比计算 ==========
        calc_frame = ttk.Frame(main_frame)
        calc_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 10))

        calc_btn_container = ttk.Frame(calc_frame)
        calc_btn_container.pack(side=tk.RIGHT, anchor="se", padx=15)

        calc_btn = tk.Button(
            calc_btn_container,
            text="优化比计算",
            command=self.calculate_ratio,
            bg="#007bff",
            fg="black",
            font=("Arial", 12, "bold"),
            activebackground="#0056b3"
        )
        calc_btn.pack(side=tk.TOP)

        self.ratio_label = ttk.Label(
            calc_btn_container,
            text="等待计算...",
            font=("YaHei", 12)
        )
        self.ratio_label.pack(side=tk.BOTTOM, pady=(5, 0))

    def select_image(self, event=None):
        """选择图片文件"""
        filetypes = [("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        path = filedialog.askopenfilename(title="选择图片", filetypes=filetypes)

        if path:
            self.image_path = path
            try:
                # 显示缩略图
                img = Image.open(path)
                # 计算缩放比例，保持宽高比
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                # 计算缩放比例
                ratio = min(canvas_width / img.width, canvas_height / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))

                # 缩放图片
                img = img.resize(new_size, Image.Resampling.LANCZOS)
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
        w, h = 300, 200  # 使用硬编码尺寸

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
            font=("YaHei", 12)
        )

    def replace_image_path_in_tvm(self, new_path):
        """替换tvm11.py文件中的图片路径"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tvm_path = os.path.join(current_dir, "tvm11.py")
        if not os.path.exists(tvm_path):
            messagebox.showerror("错误", "tvm11.py 文件不存在")
            return
        try:
            with open(tvm_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 查找包含 Image.open 的行
            new_lines = []
            replaced = False
            for line in lines:
                if "Image.open" in line:
                    # 替换路径，保留原始代码格式
                    line = re.sub(
                        r'Image\.open\(["\'][^"\']*["\']\)',
                        f'Image.open("{new_path}")',
                        line
                    )
                    replaced = True
                new_lines.append(line)

            if not replaced:
                self.log_text.insert(tk.END, "警告: 未找到 Image.open 语句\n")
                return False

            with open(tvm_path, 'w', encoding='utf-8') as file:
                file.writelines(new_lines)
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
                if "HNU-R-DDR.py" in cmd:
                    start_time = time()
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    cwd=current_dir,
                )
                self.log_text.insert(tk.END, result.stdout + "\n")
                self.update()

                # 解析 HNU-R-DDR.py 的输出
                if "HNU-R-DDR.py" in cmd:
                    probability = self.parse_probability_output(result.stdout)

            # 计算执行时间
            exec_time = round(time() - start_time, 2)

            # 添加到结果表格
            if optimize:
                item_id = self.optimized_table.insert("", "end",
                                                      values=("☐",  # 初始为未选中
                                                              self.record_id,
                                                              probability,
                                                              f"{exec_time:.2f}"),
                                                      tags=("record", "unchecked"))
                self.selection_state['optimized'][item_id] = False
            else:
                item_id = self.unoptimized_table.insert("", "end",
                                                        values=("☐",  # 初始为未选中
                                                                self.record_id,
                                                                probability,
                                                                f"{exec_time:.2f}"),
                                                        tags=("record", "unchecked"))
                self.selection_state['unoptimized'][item_id] = False

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

    def on_cell_click(self, event):
        """处理单元格点击事件"""
        # 确定点击的是哪个表格
        table = event.widget
        table_id = 'optimized' if table == self.optimized_table else 'unoptimized'

        # 获取点击的行和列
        region = table.identify_region(event.x, event.y)
        if region != 'cell':
            return

        column = table.identify_column(event.x)
        item = table.identify_row(event.y)

        # 只处理"选择"列的点击
        if column == '#1':
            # 切换选择状态
            current_state = self.selection_state[table_id].get(item, False)
            new_state = not current_state

            # 更新状态
            self.selection_state[table_id][item] = new_state
            table.set(item, "选择", "☑" if new_state else "☐")

            # 应用tag
            tags = list(table.item(item, 'tags'))
            if new_state:
                if 'unchecked' in tags:
                    tags.remove('unchecked')
                tags.append('checked')
            else:
                if 'checked' in tags:
                    tags.remove('checked')
                tags.append('unchecked')
            table.item(item, tags=tags)

            # 检查选择数量
            self.check_selection_limit()

    def check_selection_limit(self):
        """检查选择数量是否超过限制"""
        total_selected = sum(len([v for v in vals.values() if v])
                             for vals in self.selection_state.values())
        if total_selected > 2:
            # 找出最后被选中的项目并取消选择
            last_selected = None
            for table_type in self.selection_state:
                for item, state in self.selection_state[table_type].items():
                    if state:
                        last_selected = (table_type, item)

            if last_selected:
                table_type, item = last_selected
                table = self.optimized_table if table_type == 'optimized' else self.unoptimized_table
                self.selection_state[table_type][item] = False
                table.set(item, "选择", "☐")
                tags = list(table.item(item, 'tags'))
                if 'checked' in tags:
                    tags.remove('checked')
                tags.append('unchecked')
                table.item(item, tags=tags)

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
        """计算优化比"""
        # 获取选中的记录
        selected = []
        for table_type in self.selection_state:
            table = self.optimized_table if table_type == 'optimized' else self.unoptimized_table
            for item_id, is_selected in self.selection_state[table_type].items():
                if is_selected:
                    values = table.item(item_id)['values']
                    selected.append({
                        'optimized': table_type == 'optimized',
                        'time': float(values[3]),  # 输出时间
                        'id': values[1]  # ID
                    })

        # 检查选择数量
        if len(selected) != 2:
            messagebox.showwarning("警告", "请选择两条记录进行比较（一条优化，一条未优化）")
            return

        # 确保一条优化一条未优化
        if selected[0]['optimized'] == selected[1]['optimized']:
            messagebox.showwarning("警告", "请选择一条优化和一条未优化的记录进行比较")
            return

        # 确定哪条是优化的
        optimized_record = selected[0] if selected[0]['optimized'] else selected[1]
        unoptimized_record = selected[1] if selected[0]['optimized'] else selected[0]

        # 计算优化比
        time_diff = unoptimized_record['time'] - optimized_record['time']
        ratio = (time_diff / unoptimized_record['time']) * 100

        result_text = f"优化比: {ratio:.2f}% (未优化: {unoptimized_record['time']:.2f}s → 优化: {optimized_record['time']:.2f}s)"
        self.ratio_label.config(text=result_text, foreground="green")


if __name__ == "__main__":
    app = App()
    app.mainloop()