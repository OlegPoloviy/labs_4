import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import os

class ImageProcessor:
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
    
    def invert(self, img=None):
        """Негатив (інверсія)"""
        if img is None:
            img = self.current_image
        if img is None:
            return None
        
        img_array = np.array(img, dtype=np.uint8)
        inverted = 255 - img_array
        self.current_image = Image.fromarray(inverted)
        return self.current_image
    
    def logarithmic(self, img=None, c=1, r_param=1):
        """Логарифмічне перетворення: s = c * ln(1 + r * pixel)
        Впроваджено масштабування так, щоб результат займав діапазон [0,255].
        Формально: s_raw = c * ln(1 + r * pixel)
        s = (s_raw / s_raw_max) * 255, де s_raw_max = c * ln(1 + r * 255)
        """
        if img is None:
            img = self.current_image
        if img is None:
            return None
        
        img_array = np.array(img, dtype=np.float32)
        # Працюємо в діапазоні [0,255] без попередньої нормалізації в [0,1]
        r = float(r_param) if r_param is not None else 1.0
        # Захист від нульового r
        if r <= 0:
            r = 1e-6

        # Обчислюємо сировинне логарифмічне значення
        s_raw = c * np.log1p(r * img_array)

        # Максимальне можливе значення s_raw для пікселя=255
        s_raw_max = c * np.log1p(r * 255.0)
        if s_raw_max > 0:
            log_transformed = (s_raw / s_raw_max) * 255.0
        else:
            log_transformed = np.zeros_like(s_raw)

        log_transformed = np.clip(log_transformed, 0, 255)
        self.current_image = Image.fromarray(log_transformed.astype(np.uint8))
        return self.current_image
    
    def inverse_logarithmic(self, img=None, c=1, r_param=1):
        """Зворотне логарифмічне перетворення для формули s = c * ln(1 + r * pixel).
        Пряме перетворення масштабувалось в [0,255] використовуючи s_raw_max = c*ln(1+r*255).
        Для відновлення пікселя:
        pixel = (exp( (s/255) * ln(1 + r*255) ) - 1) / r
        """
        if img is None:
            img = self.current_image
        if img is None:
            return None
        
        img_array = np.array(img, dtype=np.float32)
        r = float(r_param) if r_param is not None else 1.0
        if r <= 0:
            r = 1e-6

        # Відношення від 0 до 1 від масштабованого виходу
        s_ratio = img_array / 255.0

        # Максимальний логарифмічний аргумент
        ln_max = np.log1p(r * 255.0)

        # Відновлюємо первинне ln(1 + r*pixel) значення
        ln_vals = s_ratio * ln_max

        # Використовуємо expm1 для підвищеної точності: exp(x)-1
        pixel_rec = (np.expm1(ln_vals)) / r

        pixel_rec = np.clip(pixel_rec, 0, 255)
        self.current_image = Image.fromarray(pixel_rec.astype(np.uint8))
        return self.current_image
    
    def power_law(self, img=None, gamma=1.0, c=1):
        """n-на степінь (степеневе перетворення)"""
        if img is None:
            img = self.current_image
        if img is None:
            return None
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        power_transformed = c * np.power(img_array, gamma)
        power_transformed = np.clip(power_transformed * 255, 0, 255)
        self.current_image = Image.fromarray(power_transformed.astype(np.uint8))
        return self.current_image
    
    def nth_root(self, img=None, n=2, c=1):
        """Корінь n-ої степені"""
        if img is None:
            img = self.current_image
        if img is None:
            return None
        
        gamma = 1.0 / n
        return self.power_law(img, gamma, c)
    
    def save_image(self, path):
        """Збереження зображення"""
        if self.current_image:
            self.current_image.save(path)
            return True
        return False


class ImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Градаційні перетворення зображень")
        self.root.geometry("1000x700")
        
        self.processor = ImageProcessor()
        self.display_image = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Фрейм для кнопок керування
        control_frame = tk.Frame(self.root, bg='#f0f0f0', padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Кнопки завантаження та збереження
        btn_load = tk.Button(control_frame, text="📂 Завантажити", command=self.load_image, 
                            bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_save = tk.Button(control_frame, text="💾 Зберегти", command=self.save_image,
                            bg='#2196F3', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        btn_save.pack(side=tk.LEFT, padx=5)
        
        btn_reset = tk.Button(control_frame, text="🔄 Скинути", command=self.reset_image,
                             bg='#FF9800', fg='white', font=('Arial', 10, 'bold'), padx=15, pady=5)
        btn_reset.pack(side=tk.LEFT, padx=5)
        
        # Фрейм для перетворень
        transform_frame = tk.LabelFrame(self.root, text="Градаційні перетворення", 
                                       font=('Arial', 11, 'bold'), padx=10, pady=10)
        transform_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Негатив
        tk.Button(transform_frame, text="Негатив", command=self.apply_invert,
                 bg='#9C27B0', fg='white', font=('Arial', 10), width=20, pady=5).pack(pady=5)
        
        # Логарифм
        tk.Label(transform_frame, text="Логарифм", font=('Arial', 10, 'bold')).pack(pady=(10,5))
        log_frame = tk.Frame(transform_frame)
        log_frame.pack(pady=5)
        tk.Label(log_frame, text="c:").pack(side=tk.LEFT)
        self.log_c = tk.Scale(log_frame, from_=0.1, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.log_c.set(1)
        self.log_c.pack(side=tk.LEFT)
        tk.Label(log_frame, text=" r:").pack(side=tk.LEFT, padx=(6,0))
        self.log_r = tk.Scale(log_frame, from_=0.001, to=5, resolution=0.001, orient=tk.HORIZONTAL, length=120)
        self.log_r.set(1)
        self.log_r.pack(side=tk.LEFT)
        tk.Button(transform_frame, text="Застосувати", command=self.apply_log,
                 bg='#3F51B5', fg='white', font=('Arial', 9), width=20).pack(pady=2)
        
        # Зворотний логарифм
        tk.Label(transform_frame, text="Зворотний логарифм", font=('Arial', 10, 'bold')).pack(pady=(10,5))
        invlog_frame = tk.Frame(transform_frame)
        invlog_frame.pack(pady=5)
        tk.Label(invlog_frame, text="c:").pack(side=tk.LEFT)
        self.invlog_c = tk.Scale(invlog_frame, from_=0.1, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.invlog_c.set(1)
        self.invlog_c.pack(side=tk.LEFT)
        tk.Label(invlog_frame, text=" r:").pack(side=tk.LEFT, padx=(6,0))
        self.invlog_r = tk.Scale(invlog_frame, from_=0.001, to=5, resolution=0.001, orient=tk.HORIZONTAL, length=120)
        self.invlog_r.set(1)
        self.invlog_r.pack(side=tk.LEFT)
        tk.Button(transform_frame, text="Застосувати", command=self.apply_invlog,
                 bg='#009688', fg='white', font=('Arial', 9), width=20).pack(pady=2)
        
        # Степінь
        tk.Label(transform_frame, text="n-на степінь", font=('Arial', 10, 'bold')).pack(pady=(10,5))
        power_frame = tk.Frame(transform_frame)
        power_frame.pack(pady=5)
        tk.Label(power_frame, text="γ:").pack(side=tk.LEFT)
        self.gamma = tk.Scale(power_frame, from_=0.1, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.gamma.set(1)
        self.gamma.pack(side=tk.LEFT)
        tk.Button(transform_frame, text="Застосувати", command=self.apply_power,
                 bg='#FF5722', fg='white', font=('Arial', 9), width=20).pack(pady=2)
        
        # Корінь
        tk.Label(transform_frame, text="Корінь n-ої степені", font=('Arial', 10, 'bold')).pack(pady=(10,5))
        root_frame = tk.Frame(transform_frame)
        root_frame.pack(pady=5)
        tk.Label(root_frame, text="n:").pack(side=tk.LEFT)
        self.root_n = tk.Scale(root_frame, from_=2, to=10, resolution=1, orient=tk.HORIZONTAL, length=150)
        self.root_n.set(2)
        self.root_n.pack(side=tk.LEFT)
        tk.Button(transform_frame, text="Застосувати", command=self.apply_root,
                 bg='#795548', fg='white', font=('Arial', 9), width=20).pack(pady=2)
        
        # Фрейм для відображення зображення
        self.image_frame = tk.Frame(self.root, bg='white')
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.image_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
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
            # Скинути збережений розмір при завантаженні нового зображення
            if hasattr(self, 'display_size'):
                delattr(self, 'display_size')
            img = self.processor.load_image(file_path)
            self.display_current_image()
            self.info_label.config(text=f"Завантажено: {os.path.basename(file_path)}")
    
    def save_image(self):
        if self.processor.current_image is None:
            messagebox.showwarning("Попередження", "Немає зображення для збереження!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            self.processor.save_image(file_path)
            messagebox.showinfo("Успіх", "Зображення збережено!")
    
    def reset_image(self):
        img = self.processor.reset_image()
        if img:
            self.display_current_image()
            self.info_label.config(text="Зображення скинуто до оригіналу")
    
    def apply_invert(self):
        if self.processor.current_image:
            self.processor.invert()
            self.display_current_image()
            self.info_label.config(text="Застосовано: Негатив")
    
    def apply_log(self):
        if self.processor.current_image:
            c = self.log_c.get()
            r = self.log_r.get()
            self.processor.logarithmic(c=c, r_param=r)
            self.display_current_image()
            self.info_label.config(text=f"Застосовано: Логарифм (c={c:.2f}, r={r:.3f})")
    
    def apply_invlog(self):
        if self.processor.current_image:
            c = self.invlog_c.get()
            r = self.invlog_r.get()
            self.processor.inverse_logarithmic(c=c, r_param=r)
            self.display_current_image()
            self.info_label.config(text=f"Застосовано: Зворотний логарифм (c={c:.2f}, r={r:.3f})")
    
    def apply_power(self):
        if self.processor.current_image:
            gamma = self.gamma.get()
            self.processor.power_law(gamma=gamma)
            self.display_current_image()
            self.info_label.config(text=f"Застосовано: n-на степінь (γ={gamma:.1f})")
    
    def apply_root(self):
        if self.processor.current_image:
            n = self.root_n.get()
            self.processor.nth_root(n=n)
            self.display_current_image()
            self.info_label.config(text=f"Застосовано: Корінь {n}-ої степені")
    
    def display_current_image(self):
        if self.processor.current_image is None:
            return
        
        # Оновити canvas для отримання актуальних розмірів
        self.canvas.update_idletasks()
        
        # Отримати розміри canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 600
            canvas_height = 500
        
        # При першому завантаженні зберегти розміри для масштабування
        if not hasattr(self, 'display_size'):
            img = self.processor.current_image.copy()
            img.thumbnail((canvas_width - 40, canvas_height - 40), Image.Resampling.LANCZOS)
            self.display_size = img.size
        
        # Масштабування зображення до збереженого розміру
        img = self.processor.current_image.copy()
        img = img.resize(self.display_size, Image.Resampling.LANCZOS)
        
        # Конвертація для Tkinter
        self.display_image = ImageTk.PhotoImage(img)
        
        # Відображення на canvas по центру
        self.canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.display_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()