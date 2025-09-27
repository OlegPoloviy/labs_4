import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter
import numpy as np
from scipy import ndimage
import os

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Обробка зображень - Лабораторна робота")
        self.root.geometry("1000x750")
        
        # Змінні для зберігання зображень
        self.original_image = None
        self.current_image = None
        self.photo_image = None
        
        # Попередньо визначені фільтри
        self.predefined_filters = {
            'Laplace': np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]]),
            'Hipass': np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]]),
            'Edge detection': np.array([[-1, -1, -1],
                                       [-1, 8, -1],
                                       [-1, -1, -1]]),
            'Sharpen': np.array([[-1, -1, -1],
                                [-1, 16, -1],
                                [-1, -1, -1]]) / 8,
            'Softening': np.array([[2, 2, 2],
                                  [2, 0, 2],
                                  [2, 2, 2]]) / 16,
            'Gaussian 3x3': np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]]) / 16,
            'Gaussian 5x5': np.array([[1, 4, 6, 4, 1],
                                     [4, 16, 24, 16, 4],
                                     [6, 24, 36, 24, 6],
                                     [4, 16, 24, 16, 4],
                                     [1, 4, 6, 4, 1]]) / 256,
                                     #Поміняти
            'Prewitt X': np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]]),
            'Prewitt Y': np.array([[1, 1, 1],
                                  [0, 0, 0],
                                  [-1, -1, -1]]),
            'Sobel X': np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]),
            'Sobel Y': np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])
        }
        
        # Створення меню
        self.create_menu()
        
        # Створення панелі інструментів
        self.create_toolbar()
        
        # Створення області для відображення зображення
        self.create_image_area()
        
        # Створення статусного рядка
        self.create_status_bar()
        
    def create_menu(self):
        """Створення головного меню"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню "Файл"
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Відкрити зображення", command=self.open_image)
        file_menu.add_command(label="Зберегти", command=self.save_image)
        file_menu.add_command(label="Зберегти як...", command=self.save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="Вихід", command=self.root.quit)
        
        # Меню "Обробка"
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Обробка", menu=process_menu)
        process_menu.add_command(label="Перетворити в напівтонове", command=self.convert_to_grayscale)
        process_menu.add_command(label="Відновити оригінал", command=self.restore_original)
        process_menu.add_separator()
        
        # Підменю фільтрів
        filter_menu = tk.Menu(process_menu, tearoff=0)
        process_menu.add_cascade(label="Фільтри", menu=filter_menu)
        
        # Фільтри виявлення країв
        edge_menu = tk.Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Виявлення країв", menu=edge_menu)
        edge_menu.add_command(label="Laplace", command=lambda: self.apply_filter('Laplace'))
        edge_menu.add_command(label="Edge detection", command=lambda: self.apply_filter('Edge detection'))
        edge_menu.add_command(label="Prewitt", command=self.apply_prewitt)
        edge_menu.add_command(label="Sobel", command=self.apply_sobel)
        
        # Фільтри підвищення чіткості
        sharp_menu = tk.Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Підвищення чіткості", menu=sharp_menu)
        sharp_menu.add_command(label="Hipass", command=lambda: self.apply_filter('Hipass'))
        sharp_menu.add_command(label="Sharpen", command=lambda: self.apply_filter('Sharpen'))
        
        # Фільтри згладжування
        smooth_menu = tk.Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Згладжування", menu=smooth_menu)
        smooth_menu.add_command(label="Gaussian 3×3", command=lambda: self.apply_filter('Gaussian 3x3'))
        smooth_menu.add_command(label="Gaussian 5×5", command=lambda: self.apply_filter('Gaussian 5x5'))
        smooth_menu.add_command(label="Softening", command=lambda: self.apply_filter('Softening'))
        
        filter_menu.add_separator()
        filter_menu.add_command(label="Користувацький фільтр", command=self.custom_filter_dialog)
        
        # Меню "Довідка"
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Довідка", menu=help_menu)
        help_menu.add_command(label="Про програму", command=self.show_about)
        
    def create_toolbar(self):
        """Створення панелі інструментів"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Кнопки панелі інструментів
        ttk.Button(toolbar, text="📁 Відкрити", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Зберегти", command=self.save_image).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(toolbar, text="⚫ В напівтонове", command=self.convert_to_grayscale).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔄 Відновити", command=self.restore_original).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Кнопки фільтрів
        filter_frame = ttk.Frame(toolbar)
        filter_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_frame, text="Фільтр:").pack(side=tk.LEFT, padx=2)
        self.filter_combo = ttk.Combobox(filter_frame, 
                                         values=list(self.predefined_filters.keys()) + ['Prewitt', 'Sobel'],
                                         width=15,
                                         state='readonly')
        self.filter_combo.pack(side=tk.LEFT, padx=2)
        self.filter_combo.set('Laplace')
        
        ttk.Button(filter_frame, text="Застосувати", command=self.apply_selected_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(filter_frame, text="⚙️ Свій фільтр", command=self.custom_filter_dialog).pack(side=tk.LEFT, padx=5)
        
        # Інформаційна панель
        self.info_frame = ttk.Frame(toolbar)
        self.info_frame.pack(side=tk.RIGHT, padx=10)
        
        self.info_label = ttk.Label(self.info_frame, text="Зображення не завантажено")
        self.info_label.pack(side=tk.LEFT)
        
    def create_image_area(self):
        """Створення області для відображення зображення"""
        # Рамка з прокруткою
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Створення Canvas з прокруткою
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray85")
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Розміщення елементів
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Налаштування розтягування
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
    def create_status_bar(self):
        """Створення статусного рядка"""
        self.status_bar = ttk.Label(self.root, text="Готово", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def custom_filter_dialog(self):
        """Діалогове вікно для створення користувацького фільтра"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Користувацький фільтр")
        dialog.geometry("500x600")
        dialog.resizable(False, False)
        
        # Вибір розміру фільтра
        size_frame = ttk.Frame(dialog)
        size_frame.pack(pady=10)
        
        ttk.Label(size_frame, text="Розмір фільтра:").pack(side=tk.LEFT, padx=5)
        size_var = tk.StringVar(value="3×3")
        size_combo = ttk.Combobox(size_frame, textvariable=size_var, 
                                  values=["3×3", "5×5", "7×7"],
                                  width=10, state='readonly')
        size_combo.pack(side=tk.LEFT, padx=5)
        
        # Контейнер для матриці фільтра
        matrix_container = ttk.Frame(dialog)
        matrix_container.pack(pady=10, fill=tk.BOTH, expand=True)
        
        matrix_frame = None
        entries = []
        
        def update_matrix_size(*args):
            nonlocal matrix_frame, entries
            
            if matrix_frame:
                matrix_frame.destroy()
            
            matrix_frame = ttk.Frame(matrix_container)
            matrix_frame.pack()
            
            size = int(size_var.get()[0])
            entries = []
            
            # Створення матриці полів вводу
            for i in range(size):
                row_entries = []
                for j in range(size):
                    entry = ttk.Entry(matrix_frame, width=8)
                    entry.grid(row=i, column=j, padx=2, pady=2)
                    # Початкове значення
                    if i == size//2 and j == size//2:
                        entry.insert(0, "1")
                    else:
                        entry.insert(0, "0")
                    row_entries.append(entry)
                entries.append(row_entries)
        
        size_combo.bind('<<ComboboxSelected>>', update_matrix_size)
        update_matrix_size()
        
        # Попередньо визначені шаблони
        template_frame = ttk.LabelFrame(dialog, text="Шаблони фільтрів")
        template_frame.pack(pady=10, fill=tk.X, padx=20)
        
        def load_template(template_name):
            if template_name in self.predefined_filters:
                kernel = self.predefined_filters[template_name]
                size = kernel.shape[0]
                
                # Оновлення розміру
                size_var.set(f"{size}×{size}")
                update_matrix_size()
                
                # Заповнення значень
                for i in range(size):
                    for j in range(size):
                        entries[i][j].delete(0, tk.END)
                        entries[i][j].insert(0, str(kernel[i, j]))
        
        # Кнопки шаблонів
        templates_grid = ttk.Frame(template_frame)
        templates_grid.pack(pady=5)
        
        row = 0
        col = 0
        for name in self.predefined_filters.keys():
            if col == 3:
                row += 1
                col = 0
            ttk.Button(templates_grid, text=name, width=15,
                      command=lambda n=name: load_template(n)).grid(row=row, column=col, padx=2, pady=2)
            col += 1
        
        # Опції нормалізації
        options_frame = ttk.LabelFrame(dialog, text="Опції")
        options_frame.pack(pady=10, fill=tk.X, padx=20)
        
        normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Автоматична нормалізація", 
                       variable=normalize_var).pack(pady=5)
        
        # Кнопки діалогу
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def apply_custom_filter():
            
            try:
                size = int(size_var.get()[0])
                kernel = np.zeros((size, size))
                
                # Зчитування значень з полів
                for i in range(size):
                    for j in range(size):
                        value = entries[i][j].get()
                        kernel[i, j] = float(value) if value else 0
                
                # Нормалізація якщо потрібно
                if normalize_var.get():
                    kernel_sum = np.sum(kernel)
                    if kernel_sum != 0:
                        kernel = kernel / kernel_sum
                
                # Застосування фільтра
                self.apply_custom_kernel(kernel)
                dialog.destroy()
                self.status_bar.config(text="Застосовано користувацький фільтр")
                
            except ValueError as e:
                messagebox.showerror("Помилка", "Введіть коректні числові значення")
        
        ttk.Button(button_frame, text="Застосувати", command=apply_custom_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Скасувати", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
    def apply_custom_kernel(self, kernel):
        """Застосування користувацького ядра фільтра"""
        if self.current_image:
            try:
                # --- КЛЮЧОВЕ ВИПРАВЛЕННЯ ---
                # Конвертація зображення в масив з плаваючою комою ДО згортки.
                # Це дозволяє зберігати від'ємні значення, що є критичним для фільтрів країв.
                img_array = np.array(self.current_image, dtype=np.float64)

                # Застосування фільтра
                if len(img_array.shape) == 2:  # Grayscale
                    filtered = ndimage.convolve(img_array, kernel)
                else:  # RGB
                    # Створюємо масив для результату з тим же типом float64
                    filtered = np.zeros_like(img_array, dtype=np.float64)
                    for i in range(img_array.shape[2]):
                        filtered[:, :, i] = ndimage.convolve(img_array[:, :, i], kernel)

                # Нормалізація результату для візуалізації.
                # Цей крок "розтягує" отримані значення (включно з від'ємними) на повний діапазон 0-255.
                min_val, max_val = np.min(filtered), np.max(filtered)
                if max_val - min_val > 0:
                    normalized = 255.0 * (filtered - min_val) / (max_val - min_val)
                else:
                    # Якщо зображення однорідне, просто використовуємо наявні значення
                    normalized = filtered

                # Конвертація назад в PIL Image.
                # Спочатку приводимо до типу uint8, безпечно відсікаючи значення.
                final_image_array = np.clip(normalized, 0, 255).astype(np.uint8)
                self.current_image = Image.fromarray(final_image_array)
                self.display_image()

            except Exception as e:
                messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра:\n{str(e)}")
    
    def apply_selected_filter(self):
        """Застосування вибраного з комбобокса фільтра"""
        filter_name = self.filter_combo.get()
        if filter_name == 'Prewitt':
            self.apply_prewitt()
        elif filter_name == 'Sobel':
            self.apply_sobel()
        else:
            self.apply_filter(filter_name)
    
    def apply_filter(self, filter_name):
        """Застосування попередньо визначеного фільтра"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
            
        if filter_name in self.predefined_filters:
            kernel = self.predefined_filters[filter_name]
            self.apply_custom_kernel(kernel)
            self.status_bar.config(text=f"Застосовано фільтр: {filter_name}")
    
    def apply_prewitt(self):
        """Застосування фільтра Прюіта"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
            
        try:
            img_array = np.array(self.current_image)
            
            # Конвертація в grayscale якщо потрібно
            if len(img_array.shape) == 3:
                img_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Застосування фільтрів Прюіта
            prewitt_x = ndimage.convolve(img_array, self.predefined_filters['Prewitt X'])
            prewitt_y = ndimage.convolve(img_array, self.predefined_filters['Prewitt Y'])
            
            # Обчислення градієнта
            prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
            prewitt = np.clip(prewitt, 0, 255).astype(np.uint8)
            
            self.current_image = Image.fromarray(prewitt)
            self.display_image()
            self.update_image_info()
            self.status_bar.config(text="Застосовано фільтр Прюіта")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра Прюіта:\n{str(e)}")
    
    def apply_sobel(self):
        """Застосування фільтра Собеля"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
            
        try:
            img_array = np.array(self.current_image)
            
            # Конвертація в grayscale якщо потрібно
            if len(img_array.shape) == 3:
                img_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Застосування фільтрів Собеля
            sobel_x = ndimage.convolve(img_array, self.predefined_filters['Sobel X'])
            sobel_y = ndimage.convolve(img_array, self.predefined_filters['Sobel Y'])
            
            # Обчислення градієнта
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel = np.clip(sobel, 0, 255).astype(np.uint8)
            
            self.current_image = Image.fromarray(sobel)
            self.display_image()
            self.update_image_info()
            self.status_bar.config(text="Застосовано фільтр Собеля")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра Собеля:\n{str(e)}")
    
    def open_image(self):
        """Відкриття зображення"""
        file_path = filedialog.askopenfilename(
            title="Виберіть зображення",
            filetypes=[
                ("Підтримувані формати", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("TIFF файли", "*.tif;*.tiff"),
                ("PNG файли", "*.png"),
                ("JPEG файли", "*.jpg;*.jpeg"),
                ("BMP файли", "*.bmp"),
                ("GIF файли", "*.gif"),
                ("Всі файли", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Завантаження зображення
                self.original_image = Image.open(file_path)
                self.current_image = self.original_image.copy()
                self.current_file_path = file_path
                
                # Відображення зображення
                self.display_image()
                
                # Оновлення інформації
                self.update_image_info()
                self.status_bar.config(text=f"Завантажено: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося відкрити зображення:\n{str(e)}")
    
    def display_image(self):
        """Відображення поточного зображення"""
        if self.current_image:
            # Отримання розмірів canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Якщо canvas ще не відображений, використовуємо стандартні розміри
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600
            
            # Масштабування зображення, якщо воно занадто велике
            img_width, img_height = self.current_image.size
            display_image = self.current_image.copy()
            
            # Обчислення коефіцієнта масштабування
            if img_width > canvas_width or img_height > canvas_height:
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                scale = min(scale_x, scale_y) * 0.9  # 0.9 для відступів
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                display_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Конвертація в PhotoImage
            self.photo_image = ImageTk.PhotoImage(display_image)
            
            # Очищення canvas і відображення зображення
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                anchor=tk.CENTER, 
                image=self.photo_image
            )
            
            # Оновлення області прокрутки
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def save_image(self):
        """Збереження зображення"""
        if self.current_image and hasattr(self, 'current_file_path'):
            try:
                self.current_image.save(self.current_file_path)
                self.status_bar.config(text=f"Збережено: {os.path.basename(self.current_file_path)}")
                messagebox.showinfo("Успіх", "Зображення успішно збережено")
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося зберегти зображення:\n{str(e)}")
        else:
            self.save_image_as()
    
    def save_image_as(self):
        """Збереження зображення з вибором місця"""
        if self.current_image:
            file_path = filedialog.asksaveasfilename(
                title="Зберегти зображення як",
                defaultextension=".png",
                filetypes=[
                    ("PNG файли", "*.png"),
                    ("JPEG файли", "*.jpg"),
                    ("TIFF файли", "*.tif"),
                    ("BMP файли", "*.bmp"),
                    ("Всі файли", "*.*")
                ]
            )
            
            if file_path:
                try:
                    # Визначення формату за розширенням
                    _, ext = os.path.splitext(file_path)
                    ext = ext.lower()
                    
                    # Збереження з відповідним форматом
                    if ext in ['.jpg', '.jpeg']:
                        # Конвертація RGBA в RGB для JPEG
                        if self.current_image.mode == 'RGBA':
                            rgb_image = Image.new('RGB', self.current_image.size, (255, 255, 255))
                            rgb_image.paste(self.current_image, mask=self.current_image.split()[3])
                            rgb_image.save(file_path, 'JPEG', quality=95)
                        else:
                            self.current_image.save(file_path, 'JPEG', quality=95)
                    else:
                        self.current_image.save(file_path)
                    
                    self.current_file_path = file_path
                    self.status_bar.config(text=f"Збережено: {os.path.basename(file_path)}")
                    messagebox.showinfo("Успіх", "Зображення успішно збережено")
                    
                except Exception as e:
                    messagebox.showerror("Помилка", f"Не вдалося зберегти зображення:\n{str(e)}")
        else:
            messagebox.showwarning("Увага", "Немає зображення для збереження")
    
    def convert_to_grayscale(self):
        """Перетворення зображення в напівтонове"""
        if self.current_image:
            try:
                # Перетворення в напівтонове
                self.current_image = self.current_image.convert('L')
                
                # Відображення оновленого зображення
                self.display_image()
                
                # Оновлення інформації
                self.update_image_info()
                self.status_bar.config(text="Зображення перетворено в напівтонове")
                
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося перетворити зображення:\n{str(e)}")
        else:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
    
    def restore_original(self):
        """Відновлення оригінального зображення"""
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.display_image()
            self.update_image_info()
            self.status_bar.config(text="Відновлено оригінальне зображення")
        else:
            messagebox.showwarning("Увага", "Немає оригінального зображення для відновлення")
    
    def update_image_info(self):
        """Оновлення інформації про зображення"""
        if self.current_image:
            width, height = self.current_image.size
            mode = self.current_image.mode
            
            mode_text = {
                'RGB': 'Кольорове (RGB)',
                'RGBA': 'Кольорове (RGBA)',
                'L': 'Напівтонове',
                'P': 'Палітра',
                '1': 'Чорно-біле'
            }.get(mode, mode)
            
            info_text = f"Розмір: {width}×{height} | Режим: {mode_text}"
            self.info_label.config(text=info_text)
        else:
            self.info_label.config(text="Зображення не завантажено")
    
    def show_about(self):
        """Відображення інформації про програму"""
        about_text = """Програма обробки зображень
        
Лабораторна робота
        
Функції:
• Просторова фільтрація зображень
• Фільтри виявлення країв (Laplace, Prewitt, Sobel)
• Фільтри згладжування (Gaussian)
• Фільтри підвищення чіткості
• Створення користувацьких фільтрів
        
Версія: 1.0
Автор:  Польовий Олег"""
        
        messagebox.showinfo("Про програму", about_text)
    
    def apply_gaussian_highpass(self):
        """Застосування високочастотного фільтра Гауса"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
        
        try:
            # Спочатку застосовуємо низькочастотний фільтр Гауса
            img_array = np.array(self.current_image)
            
            # Великий фільтр Гауса для розмиття
            gaussian_kernel = np.array([[1, 4, 7, 4, 1],
                                       [4, 16, 26, 16, 4],
                                       [7, 26, 41, 26, 7],
                                       [4, 16, 26, 16, 4],
                                       [1, 4, 7, 4, 1]]) / 273
            
            if len(img_array.shape) == 2:
                blurred = ndimage.convolve(img_array, gaussian_kernel)
                # Високочастотний = Оригінал - Низькочастотний
                highpass = img_array - blurred + 128
            else:
                blurred = np.zeros_like(img_array)
                highpass = np.zeros_like(img_array)
                for i in range(img_array.shape[2]):
                    blurred[:, :, i] = ndimage.convolve(img_array[:, :, i], gaussian_kernel)
                    highpass[:, :, i] = img_array[:, :, i] - blurred[:, :, i] + 128
            
            highpass = np.clip(highpass, 0, 255).astype(np.uint8)
            self.current_image = Image.fromarray(highpass)
            self.display_image()
            self.update_image_info()
            self.status_bar.config(text="Застосовано високочастотний фільтр Гауса")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра:\n{str(e)}")
    
    def apply_laplacian_lowpass(self):
        """Застосування низькочастотного фільтра на основі Лапласа"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
        
        try:
            img_array = np.array(self.current_image)
            
            # Інвертований фільтр Лапласа для згладжування
            lowpass_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]]) / 1
            
            if len(img_array.shape) == 2:
                filtered = ndimage.convolve(img_array.astype(float), lowpass_kernel)
            else:
                filtered = np.zeros_like(img_array, dtype=float)
                for i in range(img_array.shape[2]):
                    filtered[:, :, i] = ndimage.convolve(img_array[:, :, i].astype(float), lowpass_kernel)
            
            filtered = np.clip(filtered, 0, 255).astype(np.uint8)
            self.current_image = Image.fromarray(filtered)
            self.display_image()
            self.update_image_info()
            self.status_bar.config(text="Застосовано низькочастотний фільтр Лапласа")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра:\n{str(e)}")
    
    def apply_laplacian_highpass(self):
        """Застосування високочастотного фільтра Лапласа"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
        
        # Використовуємо стандартний фільтр Лапласа як високочастотний
        self.apply_filter('Laplace')
    
    def apply_batch_filters(self):
        """Діалог для застосування декількох фільтрів послідовно"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Пакетна обробка фільтрами")
        dialog.geometry("400x500")
        
        ttk.Label(dialog, text="Виберіть фільтри для послідовного застосування:",
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Список доступних фільтрів
        filters_frame = ttk.Frame(dialog)
        filters_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Створення списку з чекбоксами
        filter_vars = {}
        all_filters = list(self.predefined_filters.keys()) + ['Prewitt', 'Sobel', 'Gaussian HighPass', 'Laplacian LowPass']
        
        canvas = tk.Canvas(filters_frame)
        scrollbar = ttk.Scrollbar(filters_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for filter_name in all_filters:
            var = tk.BooleanVar()
            filter_vars[filter_name] = var
            ttk.Checkbutton(scrollable_frame, text=filter_name, variable=var).pack(anchor='w', padx=10, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Кнопки
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def apply_selected():
            selected_filters = [name for name, var in filter_vars.items() if var.get()]
            
            if not selected_filters:
                messagebox.showwarning("Увага", "Виберіть хоча б один фільтр")
                return
            
            try:
                # Застосування фільтрів послідовно
                for filter_name in selected_filters:
                    if filter_name == 'Prewitt':
                        self.apply_prewitt()
                    elif filter_name == 'Sobel':
                        self.apply_sobel()
                    elif filter_name == 'Gaussian HighPass':
                        self.apply_gaussian_highpass()
                    elif filter_name == 'Laplacian LowPass':
                        self.apply_laplacian_lowpass()
                    else:
                        self.apply_filter(filter_name)
                
                dialog.destroy()
                self.status_bar.config(text=f"Застосовано {len(selected_filters)} фільтрів")
                messagebox.showinfo("Успіх", f"Успішно застосовано {len(selected_filters)} фільтрів")
                
            except Exception as e:
                messagebox.showerror("Помилка", f"Помилка при застосуванні фільтрів:\n{str(e)}")
        
        ttk.Button(button_frame, text="Застосувати", command=apply_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Скасувати", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def compare_images(self):
        """Порівняння оригінального і обробленого зображень"""
        if not self.original_image or not self.current_image:
            messagebox.showwarning("Увага", "Потрібно мати оригінальне і оброблене зображення")
            return
        
        # Створення нового вікна для порівняння
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Порівняння зображень")
        compare_window.geometry("1200x700")
        
        # Рамки для зображень
        left_frame = ttk.LabelFrame(compare_window, text="Оригінал")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = ttk.LabelFrame(compare_window, text="Оброблене")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas для оригінального зображення
        left_canvas = tk.Canvas(left_frame, bg="gray85")
        left_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Canvas для обробленого зображення
        right_canvas = tk.Canvas(right_frame, bg="gray85")
        right_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Масштабування і відображення зображень
        def display_comparison():
            # Отримання розмірів canvas
            canvas_width = 550
            canvas_height = 600
            
            # Оригінальне зображення
            img_width, img_height = self.original_image.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y) * 0.9
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Відображення оригіналу
            original_resized = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            original_photo = ImageTk.PhotoImage(original_resized)
            left_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=original_photo)
            left_canvas.image = original_photo  # Зберігаємо посилання
            
            # Відображення обробленого
            current_resized = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            current_photo = ImageTk.PhotoImage(current_resized)
            right_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=current_photo)
            right_canvas.image = current_photo  # Зберігаємо посилання
        
        # Відображення після створення вікна
        compare_window.after(100, display_comparison)
    
    def export_filter_matrix(self):
        """Експорт поточної матриці фільтра у файл"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Експорт матриці фільтра")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Виберіть фільтр для експорту:").pack(pady=10)
        
        # Список фільтрів
        filter_listbox = tk.Listbox(dialog, height=10)
        filter_listbox.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        for name in self.predefined_filters.keys():
            filter_listbox.insert(tk.END, name)
        
        def export_selected():
            selection = filter_listbox.curselection()
            if not selection:
                messagebox.showwarning("Увага", "Виберіть фільтр")
                return
            
            filter_name = filter_listbox.get(selection[0])
            kernel = self.predefined_filters[filter_name]
            
            file_path = filedialog.asksaveasfilename(
                title="Зберегти матрицю фільтра",
                defaultextension=".txt",
                filetypes=[("Text файли", "*.txt"), ("CSV файли", "*.csv")]
            )
            
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        f.write(f"# Фільтр: {filter_name}\n")
                        f.write(f"# Розмір: {kernel.shape[0]}x{kernel.shape[1]}\n\n")
                        for row in kernel:
                            f.write('\t'.join([str(val) for val in row]) + '\n')
                    
                    messagebox.showinfo("Успіх", "Матрицю фільтра експортовано")
                    dialog.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Помилка", f"Помилка експорту:\n{str(e)}")
        
        ttk.Button(dialog, text="Експортувати", command=export_selected).pack(pady=10)
        ttk.Button(dialog, text="Скасувати", command=dialog.destroy).pack()


# Головна функція запуску програми
def main():
    root = tk.Tk()
    app = ImageProcessor(root)
    
    # Центрування вікна на екрані
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()