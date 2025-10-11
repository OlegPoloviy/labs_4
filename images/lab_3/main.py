import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

class HistogramProcessor:
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.image_path = None
    
    def load_image(self, path):
        """Завантаження зображення"""
        self.image_path = path
        self.original_image = Image.open(path).convert('RGB')
        self.current_image = self.original_image.copy()
        return self.current_image
    
    def reset_image(self):
        """Скидання до оригінального зображення"""
        if self.original_image:
            self.current_image = self.original_image.copy()
            return self.current_image
        return None
    
    def get_histogram(self, img=None):
        """Отримання гістограми зображення"""
        if img is None:
            img = self.current_image
        if img is None:
            return None, None, None
        
        img_array = np.array(img)
        
        # Гістограми для кожного каналу
        hist_r = np.histogram(img_array[:,:,0], bins=256, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:,:,1], bins=256, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:,:,2], bins=256, range=(0, 256))[0]
        
        return hist_r, hist_g, hist_b
    
    def histogram_equalization(self, img=None):
        """Еквалізація гістограми"""
        if img is None:
            img = self.current_image
        if img is None:
            return None
        
        img_array = np.array(img)
        equalized = np.zeros_like(img_array)
        
        # Еквалізація для кожного каналу окремо
        for i in range(3):
            channel = img_array[:,:,i]
            hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
            
            # Обчислення кумулятивної функції розподілу
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            
            # Інтерполяція значень
            equalized[:,:,i] = np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape)
        
        self.current_image = Image.fromarray(equalized.astype(np.uint8))
        return self.current_image, cdf_normalized
    
    def power_law_transform(self, gamma=1.0):
        """Степеневе перетворення"""
        if self.current_image is None:
            return None
        
        img_array = np.array(self.current_image, dtype=np.float32) / 255.0
        transformed = np.power(img_array, gamma) * 255
        self.current_image = Image.fromarray(transformed.astype(np.uint8))
        
        # Функція перетворення для графіка
        x = np.linspace(0, 255, 256)
        y = np.power(x / 255.0, gamma) * 255
        
        return self.current_image, x, y
    
    def logarithmic_transform(self, c=1.0):
        """Логарифмічне перетворення"""
        if self.current_image is None:
            return None
        
        img_array = np.array(self.current_image, dtype=np.float32)
        img_normalized = img_array / 255.0
        
        transformed = c * np.log(1 + img_normalized * (np.e - 1)) * 255
        self.current_image = Image.fromarray(transformed.astype(np.uint8))
        
        # Функція перетворення для графіка
        x = np.linspace(0, 255, 256)
        y = c * np.log(1 + (x / 255.0) * (np.e - 1)) * 255
        
        return self.current_image, x, y
    
    def save_image(self, path):
        """Збереження зображення"""
        if self.current_image:
            self.current_image.save(path)
            return True
        return False


class HistogramGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Обробка гістограм зображень")
        self.root.geometry("1400x900")
        
        self.processor = HistogramProcessor()
        self.original_display = None
        self.processed_display = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Верхня панель керування
        control_frame = tk.Frame(self.root, bg='#f0f0f0', padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        btn_load = tk.Button(control_frame, text="📂 Завантажити", command=self.load_image, 
                            bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_save = tk.Button(control_frame, text="💾 Зберегти", command=self.save_image,
                            bg='#2196F3', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        btn_save.pack(side=tk.LEFT, padx=5)
        
        btn_reset = tk.Button(control_frame, text="🔄 Скинути", command=self.reset_image,
                             bg='#FF9800', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        btn_reset.pack(side=tk.LEFT, padx=5)
        
        btn_save_report = tk.Button(control_frame, text="📊 Зберегти звіт", command=self.save_report,
                                    bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        btn_save_report.pack(side=tk.LEFT, padx=5)
        
        # Головний контейнер
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Ліва панель - перетворення
        left_panel = tk.LabelFrame(main_container, text="Методи обробки", 
                                  font=('Arial', 11, 'bold'), padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Еквалізація гістограми
        tk.Label(left_panel, text="Еквалізація гістограми", 
                font=('Arial', 10, 'bold')).pack(pady=(5,5))
        tk.Button(left_panel, text="Застосувати еквалізацію", command=self.apply_equalization,
                 bg='#3F51B5', fg='white', font=('Arial', 9), width=25, pady=5).pack(pady=5)
        
        tk.Label(left_panel, text="─" * 30).pack(pady=10)
        
        # Степеневе перетворення
        tk.Label(left_panel, text="Степеневе перетворення (γ)", 
                font=('Arial', 10, 'bold')).pack(pady=(5,5))
        
        gamma_frame = tk.Frame(left_panel)
        gamma_frame.pack(pady=5)
        tk.Label(gamma_frame, text="γ:").pack(side=tk.LEFT)
        self.gamma_scale = tk.Scale(gamma_frame, from_=0.1, to=3.0, resolution=0.1, 
                                    orient=tk.HORIZONTAL, length=150)
        self.gamma_scale.set(1.0)
        self.gamma_scale.pack(side=tk.LEFT)
        
        tk.Button(left_panel, text="Застосувати степінь", command=self.apply_power,
                 bg='#FF5722', fg='white', font=('Arial', 9), width=25, pady=5).pack(pady=5)
        
        tk.Label(left_panel, text="─" * 30).pack(pady=10)
        
        # Логарифмічне перетворення
        tk.Label(left_panel, text="Логарифмічне перетворення", 
                font=('Arial', 10, 'bold')).pack(pady=(5,5))
        
        log_frame = tk.Frame(left_panel)
        log_frame.pack(pady=5)
        tk.Label(log_frame, text="c:").pack(side=tk.LEFT)
        self.log_scale = tk.Scale(log_frame, from_=0.5, to=2.0, resolution=0.1, 
                                 orient=tk.HORIZONTAL, length=150)
        self.log_scale.set(1.0)
        self.log_scale.pack(side=tk.LEFT)
        
        tk.Button(left_panel, text="Застосувати логарифм", command=self.apply_log,
                 bg='#009688', fg='white', font=('Arial', 9), width=25, pady=5).pack(pady=5)
        
        # Права панель - відображення
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Верхня частина - зображення
        images_frame = tk.Frame(right_panel)
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Оригінальне зображення
        original_frame = tk.LabelFrame(images_frame, text="Оригінальне зображення", 
                                      font=('Arial', 10, 'bold'))
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(original_frame, bg='white', height=250)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Оброблене зображення
        processed_frame = tk.LabelFrame(images_frame, text="Оброблене зображення", 
                                       font=('Arial', 10, 'bold'))
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.processed_canvas = tk.Canvas(processed_frame, bg='white', height=250)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Нижня частина - графіки
        plots_frame = tk.Frame(right_panel)
        plots_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Контейнер для matplotlib графіків
        self.figure = Figure(figsize=(14, 5))
        self.canvas_plot = FigureCanvasTkAgg(self.figure, plots_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Інформаційна мітка
        self.info_label = tk.Label(self.root, text="Завантажте зображення для початку роботи", 
                                   font=('Arial', 10), bg='#f0f0f0')
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Виберіть зображення",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.processor.load_image(file_path)
            self.update_display()
            self.info_label.config(text=f"Завантажено: {os.path.basename(file_path)}")
    
    def save_image(self):
        if self.processor.current_image is None:
            messagebox.showwarning("Попередження", "Немає зображення для збереження!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        if file_path:
            self.processor.save_image(file_path)
            messagebox.showinfo("Успіх", "Зображення збережено!")
    
    def save_report(self):
        """Збереження повного звіту"""
        if self.processor.original_image is None:
            messagebox.showwarning("Попередження", "Немає даних для збереження!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        if file_path:
            self.figure.savefig(file_path, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Успіх", "Звіт збережено!")
    
    def reset_image(self):
        self.processor.reset_image()
        self.update_display()
        self.info_label.config(text="Зображення скинуто до оригіналу")
    
    def apply_equalization(self):
        if self.processor.original_image is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте зображення!")
            return
        
        self.processor.reset_image()
        result = self.processor.histogram_equalization()
        if result:
            self.update_display(show_transform=True, transform_type="equalization")
            self.info_label.config(text="Застосовано: Еквалізація гістограми")
    
    def apply_power(self):
        if self.processor.original_image is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте зображення!")
            return
        
        self.processor.reset_image()
        gamma = self.gamma_scale.get()
        result = self.processor.power_law_transform(gamma)
        if result:
            self.update_display(show_transform=True, transform_type="power", 
                              transform_data=result[1:])
            self.info_label.config(text=f"Застосовано: Степеневе перетворення (γ={gamma:.1f})")
    
    def apply_log(self):
        if self.processor.original_image is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте зображення!")
            return
        
        self.processor.reset_image()
        c = self.log_scale.get()
        result = self.processor.logarithmic_transform(c)
        if result:
            self.update_display(show_transform=True, transform_type="log", 
                              transform_data=result[1:])
            self.info_label.config(text=f"Застосовано: Логарифмічне перетворення (c={c:.1f})")
    
    def update_display(self, show_transform=False, transform_type=None, transform_data=None):
        """Оновлення відображення"""
        # Відображення оригінального зображення
        if self.processor.original_image:
            self.display_image(self.processor.original_image, self.original_canvas, "original")
        
        # Відображення обробленого зображення
        if self.processor.current_image:
            self.display_image(self.processor.current_image, self.processed_canvas, "processed")
        
        # Оновлення графіків
        self.update_plots(show_transform, transform_type, transform_data)
    
    def display_image(self, img, canvas, img_type):
        """Відображення зображення на canvas"""
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 300
        if canvas_height <= 1:
            canvas_height = 250
        
        # Масштабування
        img_copy = img.copy()
        img_copy.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
        
        # Конвертація для Tkinter
        if img_type == "original":
            self.original_display = ImageTk.PhotoImage(img_copy)
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, 
                              anchor=tk.CENTER, image=self.original_display)
        else:
            self.processed_display = ImageTk.PhotoImage(img_copy)
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, 
                              anchor=tk.CENTER, image=self.processed_display)
    
    def update_plots(self, show_transform=False, transform_type=None, transform_data=None):
        """Оновлення графіків"""
        self.figure.clear()
        
        if self.processor.original_image is None:
            return
        
        if show_transform and transform_type:
            # 4 графіки: оригінальна гістограма, функція перетворення, 
            # оброблена гістограма, порівняння
            gs = self.figure.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            ax1 = self.figure.add_subplot(gs[0, 0])
            ax2 = self.figure.add_subplot(gs[0, 1])
            ax3 = self.figure.add_subplot(gs[0, 2])
            ax4 = self.figure.add_subplot(gs[1, :])
        else:
            # Тільки оригінальна гістограма
            ax1 = self.figure.add_subplot(1, 1, 1)
        
        # Оригінальна гістограма
        hist_r, hist_g, hist_b = self.processor.get_histogram(self.processor.original_image)
        ax1.plot(hist_r, color='red', alpha=0.7, label='R', linewidth=1.5)
        ax1.plot(hist_g, color='green', alpha=0.7, label='G', linewidth=1.5)
        ax1.plot(hist_b, color='blue', alpha=0.7, label='B', linewidth=1.5)
        ax1.set_title('Гістограма оригінального зображення', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Яскравість')
        ax1.set_ylabel('Кількість пікселів')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if show_transform and transform_type:
            # Функція перетворення
            if transform_type == "equalization":
                # Для еквалізації показуємо CDF
                img_gray = np.array(self.processor.original_image.convert('L'))
                hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = cdf * 255 / cdf[-1]
                
                ax2.plot(range(256), cdf_normalized, color='purple', linewidth=2)
                ax2.set_title('Функція еквалізації (CDF)', fontsize=10, fontweight='bold')
            else:
                # Для інших перетворень показуємо функцію
                x, y = transform_data
                ax2.plot(x, y, color='purple', linewidth=2)
                ax2.plot([0, 255], [0, 255], 'k--', alpha=0.3, label='y=x')
                ax2.legend()
                title = 'Функція степеневого перетворення' if transform_type == "power" else 'Функція логарифмічного перетворення'
                ax2.set_title(title, fontsize=10, fontweight='bold')
            
            ax2.set_ylabel('Вихідна яскравість')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, 255])
            ax2.set_ylim([0, 255])
            
            # Гістограма обробленого зображення
            hist_r2, hist_g2, hist_b2 = self.processor.get_histogram(self.processor.current_image)
            ax3.plot(hist_r2, color='red', alpha=0.7, label='R', linewidth=1.5)
            ax3.plot(hist_g2, color='green', alpha=0.7, label='G', linewidth=1.5)
            ax3.plot(hist_b2, color='blue', alpha=0.7, label='B', linewidth=1.5)
            ax3.set_title('Гістограма обробленого зображення', fontsize=10, fontweight='bold')
            ax3.set_xlabel('Яскравість')
            ax3.set_ylabel('Кількість пікселів')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Порівняння гістограм (сумарна для всіх каналів)
            hist_orig_total = hist_r + hist_g + hist_b
            hist_proc_total = hist_r2 + hist_g2 + hist_b2
            
            ax4.plot(hist_orig_total, color='blue', alpha=0.6, label='Оригінал', linewidth=2)
            ax4.plot(hist_proc_total, color='red', alpha=0.6, label='Оброблено', linewidth=2)
            ax4.set_title('Порівняння гістограм', fontsize=10, fontweight='bold')
            ax4.set_xlabel('Яскравість')
            ax4.set_ylabel('Кількість пікселів')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        self.canvas_plot.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = HistogramGUI(root)
    root.mainloop()