import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from scipy import ndimage
import os

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Обробка зображень - Лабораторна робота")
        self.root.geometry("1400x800")

        # Змінні для зберігання зображень
        self.original_image = None
        self.current_image = None
        self.original_photo_image = None
        self.processed_photo_image = None
        self.current_file_path = None

        # Попередньо визначені фільтри
        self.predefined_filters = {
            'Laplace': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
            'Hipass': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
            'Edge detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            'Sharpen': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
            'Softening': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,  # Виправлено: правильна нормалізація
            'Gaussian 3x3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            'Gaussian 5x5': np.array([[1, 4, 6, 4, 1],
                                      [4, 16, 24, 16, 4],
                                      [6, 24, 36, 24, 6],
                                      [4, 16, 24, 16, 4],
                                      [1, 4, 6, 4, 1]]) / 256,
            'Prewitt X': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            'Prewitt Y': np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
            'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'Sobel Y': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        }

        # Створення GUI
        self.create_menu()
        self.create_toolbar()
        self.create_filter_matrix_display()
        self.create_image_area()
        self.create_status_bar()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Відкрити зображення", command=self.open_image)
        file_menu.add_command(label="Зберегти оброблене", command=self.save_image)
        file_menu.add_command(label="Зберегти оброблене як...", command=self.save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="Вихід", command=self.root.quit)

        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Обробка", menu=process_menu)
        process_menu.add_command(label="Перетворити в напівтонове", command=self.convert_to_grayscale)
        process_menu.add_command(label="Відновити оригінал", command=self.restore_original)
        process_menu.add_separator()
        process_menu.add_command(label="Користувацький фільтр", command=self.custom_filter_dialog)

    def create_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="📁 Відкрити", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Зберегти", command=self.save_image).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(toolbar, text="⚫️ В напівтонове", command=self.convert_to_grayscale).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔄 Відновити", command=self.restore_original).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)


        filter_frame = ttk.Frame(toolbar)
        filter_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(filter_frame, text="Фільтр:").pack(side=tk.LEFT, padx=2)
        self.filter_combo = ttk.Combobox(filter_frame, values=list(self.predefined_filters.keys()) + ['Prewitt', 'Sobel'], width=15, state='readonly')
        self.filter_combo.pack(side=tk.LEFT, padx=2)
        self.filter_combo.set('Laplace')
        ttk.Button(filter_frame, text="Застосувати", command=self.apply_selected_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(filter_frame, text="⚙️ Свій фільтр", command=self.custom_filter_dialog).pack(side=tk.LEFT, padx=5)

        self.info_frame = ttk.Frame(toolbar)
        self.info_frame.pack(side=tk.RIGHT, padx=10)
        self.info_label = ttk.Label(self.info_frame, text="Зображення не завантажено")
        self.info_label.pack(side=tk.LEFT)

    def create_filter_matrix_display(self):
        """Створення окремого рядка для відображення матриці фільтра"""
        # Основний фрейм для матриці фільтра
        self.matrix_display_frame = ttk.LabelFrame(self.root, text="Поточний фільтр")
        self.matrix_display_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Контейнер для матриці (аналогічно до custom_filter_dialog)
        self.matrix_container = ttk.Frame(self.matrix_display_frame)
        self.matrix_container.pack(pady=5, fill=tk.X, expand=True)
        
        # Фрейм для матриці (буде оновлюватися динамічно)
        self.matrix_frame = ttk.Frame(self.matrix_container)
        self.matrix_frame.pack()
        
        # Початковий стан
        self.matrix_entries = []
        self.current_matrix_size = 0
        
        # Заголовок для початкового стану
        self.matrix_title_label = ttk.Label(self.matrix_display_frame, text="Фільтр не застосовано", font=('Arial', 10))
        self.matrix_title_label.pack(pady=5)

    def create_image_area(self):
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.LabelFrame(paned_window, text="Оригінальне зображення")
        self.original_canvas = tk.Canvas(left_frame, bg="gray85")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        paned_window.add(left_frame, weight=1)

        right_frame = ttk.LabelFrame(paned_window, text="Оброблене зображення")
        self.processed_canvas = tk.Canvas(right_frame, bg="gray85")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        paned_window.add(right_frame, weight=1)

    def create_status_bar(self):
        self.status_bar = ttk.Label(self.root, text="Готово", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Виберіть зображення",
            filetypes=[("Image Files", "*.jpeg *.jpg *.png *.tiff *.tif *.bmp *.gif"), ("All files", "*.*")]
        )
        if not file_path: return

        try:
            self.original_image = Image.open(file_path)
            self.current_image = self.original_image.copy()
            self.current_file_path = file_path

            self.display_original_image()
            self.display_processed_image()

            self.update_image_info()
            self.status_bar.config(text=f"Завантажено: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося відкрити зображення:\n{e}")

    def _display_image_on_canvas(self, image_to_display, canvas, photo_image_attr):
        if not image_to_display: return

        self.root.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1: canvas_width = 600
        if canvas_height <= 1: canvas_height = 600

        img_width, img_height = image_to_display.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.98


        display_image = image_to_display.resize((int(img_width * scale), int(img_height * scale)), Image.Resampling.LANCZOS) if scale < 1.0 else image_to_display

        photo_image = ImageTk.PhotoImage(display_image)
        setattr(self, photo_image_attr, photo_image)

        canvas.delete("all")
        canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=photo_image)

    def display_original_image(self):
        self._display_image_on_canvas(self.original_image, self.original_canvas, 'original_photo_image')

    def display_processed_image(self):
        self._display_image_on_canvas(self.current_image, self.processed_canvas, 'processed_photo_image')

    def apply_custom_kernel(self, kernel):
        """
        Застосування користувацького ядра фільтра з коректною обробкою режимів зображення та нормалізацією.
        """
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return

        try:
            # Створюємо робочу копію зображення
            image_to_process = self.current_image.copy()

            # КРОК 1: ВИЗНАЧЕННЯ ТИПУ ФІЛЬТРА
            # Якщо в ядрі є від'ємні значення (як у Sobel, Laplace), це, ймовірно,
            # фільтр для виділення контурів. Такі фільтри найкраще працюють з одним каналом яскравості.
            is_edge_detection_filter = np.any(kernel < 0)

            # КРОК 2: ПЕРЕТВОРЕННЯ В НАПІВТОНОВЕ (ЯКЩО ПОТРІБНО)
            # Це ключове виправлення. Фільтри контурів працюють з яскравістю, не з кольором.
            if is_edge_detection_filter and image_to_process.mode != 'L':
                image_to_process = image_to_process.convert('L')

            # Конвертуємо зображення в масив NumPy для обчислень
            img_array = np.array(image_to_process, dtype=np.float64)

            # КРОК 3: ЗАСТОСУВАННЯ ФІЛЬТРА (ЗГОРТКА)
            if img_array.ndim == 2:  # Напівтонове зображення (один канал)
                filtered = ndimage.convolve(img_array, kernel)
            else:  # Кольорове зображення (тільки для фільтрів без від'ємних значень, як розмиття)
                filtered = np.zeros_like(img_array, dtype=np.float64)
                for i in range(img_array.shape[2]):  # Застосовуємо до кожного каналу (R, G, B)
                    filtered[:, :, i] = ndimage.convolve(img_array[:, :, i], kernel)

            # КРОК 4: КОРЕКТНА НОРМАЛІЗАЦІЯ РЕЗУЛЬТАТУ
            # Це найважливіша частина виправлення
            if is_edge_detection_filter:
                # Для фільтрів контурів ми хочемо бачити абсолютну величину змін яскравості.
                # Це дозволяє побачити і світлі, і темні контури.
                filtered_abs = np.abs(filtered)
                # Розтягуємо діапазон значень до 0-255 для візуалізації.
                if np.max(filtered_abs) > 0:
                    final_array = (255.0 * filtered_abs / np.max(filtered_abs)).astype(np.uint8)
                else:
                    final_array = np.zeros_like(filtered, dtype=np.uint8)
            else:
                # Для інших фільтрів (наприклад, розмиття) ми просто обрізаємо значення,
                # щоб вони залишалися в діапазоні 0-255. Це працює правильно, якщо
                # сума ядра = 1 (як у 'Softening' або 'Gaussian').
                final_array = np.clip(filtered, 0, 255).astype(np.uint8)

            # Оновлюємо поточне зображення
            self.current_image = Image.fromarray(final_array)
            self.display_processed_image()
            self.update_image_info()

        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра:\n{str(e)}")


    
    def apply_selected_filter(self):
        filter_name = self.filter_combo.get()
        if filter_name == 'Prewitt': self.apply_prewitt()
        elif filter_name == 'Sobel': self.apply_sobel()
        else: self.apply_filter(filter_name)

    def apply_filter(self, filter_name):
        if not self.current_image: messagebox.showwarning("Увага", "Спочатку відкрийте зображення"); return
        if filter_name in self.predefined_filters:
            self.apply_custom_kernel(self.predefined_filters[filter_name])
            self.status_bar.config(text=f"Застосовано фільтр: {filter_name}")

    def apply_gradient_filter(self, kernel_x, kernel_y, filter_name):
        if not self.current_image: 
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
        try:
            img_array = np.array(self.current_image.convert('L'), dtype=np.float64)
            grad_x = ndimage.convolve(img_array, kernel_x)
            grad_y = ndimage.convolve(img_array, kernel_y)
            
            # Обчислення градієнта (величина вектора градієнта)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            
            # Нормалізація до діапазону 0-255
            if np.max(gradient) > 0:
                gradient = (255.0 * gradient / np.max(gradient)).astype(np.uint8)
            else:
                gradient = np.zeros_like(gradient, dtype=np.uint8)
            
            self.current_image = Image.fromarray(gradient)
            self.display_processed_image()
            self.update_image_info()
            self.status_bar.config(text=f"Застосовано фільтр {filter_name}")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра {filter_name}:\n{e}")

    def apply_prewitt(self):
        self.apply_gradient_filter(self.predefined_filters['Prewitt X'], self.predefined_filters['Prewitt Y'], 'Прюіта')

    def apply_sobel(self):
        self.apply_gradient_filter(self.predefined_filters['Sobel X'], self.predefined_filters['Sobel Y'], 'Собеля')


    def save_image(self):
        self.save_image_as(self.current_file_path) if self.current_image and self.current_file_path else self.save_image_as()
            
    def save_image_as(self, file_path=None):
        if not self.current_image: messagebox.showwarning("Увага", "Немає зображення для збереження"); return

        if file_path is None:
            file_path = filedialog.asksaveasfilename(title="Зберегти зображення як", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        
        if not file_path: return

        try:
            image_to_save = self.current_image
            if file_path.lower().endswith(('.jpg', '.jpeg')) and image_to_save.mode == 'RGBA':
                rgb_image = Image.new("RGB", image_to_save.size, (255, 255, 255))
                rgb_image.paste(image_to_save, mask=image_to_save.split()[3])
                image_to_save = rgb_image

            image_to_save.save(file_path)
            self.current_file_path = file_path
            self.status_bar.config(text=f"Збережено: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося зберегти зображення:\n{e}")

    def convert_to_grayscale(self):
        if self.current_image:
            self.current_image = self.current_image.convert('L')
            # --- ВИПРАВЛЕННЯ: Викликаємо правильну функцію ---
            self.display_processed_image()
            self.update_image_info()
            self.status_bar.config(text="Зображення перетворено в напівтонове")
        else:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")

    def restore_original(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            # --- ВИПРАВЛЕННЯ: Викликаємо правильну функцію ---
            self.display_processed_image()
            self.update_image_info()
            self.status_bar.config(text="Відновлено оригінальне зображення")
        else:
            messagebox.showwarning("Увага", "Немає оригінального зображення для відновлення")


    def update_image_info(self):
        if self.current_image:
            mode_map = {'RGB': 'Кольорове (RGB)', 'RGBA': 'Кольорове (RGBA)', 'L': 'Напівтонове'}
            mode_text = mode_map.get(self.current_image.mode, self.current_image.mode)
            self.info_label.config(text=f"Розмір: {self.current_image.width}x{self.current_image.height} | Режим: {mode_text}")
        else:
            self.info_label.config(text="Зображення не завантажено")
        
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
                self.update_filter_matrix_display("Користувацький", kernel)
                dialog.destroy()
                self.status_bar.config(text="Застосовано користувацький фільтр")
                
            except ValueError as e:
                messagebox.showerror("Помилка", "Введіть коректні числові значення")
        
        ttk.Button(button_frame, text="Застосувати", command=apply_custom_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Скасувати", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        def apply_custom_kernel(self, kernel):
            """
            Застосування користувацького ядра фільтра з коректною обробкою режимів зображення.
            """
            if not self.current_image:
                messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
                return

            try:
                # Створюємо робочу копію зображення
                image_to_process = self.current_image.copy()

                # КРОК 1: ВИЗНАЧЕННЯ ТИПУ ФІЛЬТРА
                # Якщо в ядрі є від'ємні значення (як у Sobel, Prewitt, Laplace),
                # то це фільтр для виділення контурів, і він повинен працювати з одним каналом яскравості.
                is_edge_detection_filter = np.any(kernel < 0)

                # КРОК 2: ПРИМУСОВА КОНВЕРТАЦІЯ В НАПІВТОНОВЕ ('L')
                # Це ключове виправлення! Воно відкине зайвий альфа-канал з режиму 'LA'.
                if is_edge_detection_filter and image_to_process.mode != 'L':
                    image_to_process = image_to_process.convert('L')

                img_array = np.array(image_to_process, dtype=np.float64)

                # КРОК 3: ЗАСТОСУВАННЯ ФІЛЬТРА (тепер тільки до одного каналу)
                if img_array.ndim == 2:
                    filtered = ndimage.convolve(img_array, kernel)
                else:  # Цей блок тепер буде виконуватися тільки для фільтрів розмиття на кольорових зображеннях
                    filtered = np.zeros_like(img_array, dtype=np.float64)
                    for i in range(img_array.shape[2]):
                        filtered[:, :, i] = ndimage.convolve(img_array[:, :, i], kernel)

                # КРОК 4: КОРЕКТНА НОРМАЛІЗАЦІЯ РЕЗУЛЬТАТУ
                if is_edge_detection_filter:
                    # Для фільтрів контурів ми візуалізуємо абсолютну величину змін яскравості.
                    filtered_abs = np.abs(filtered)
                    if np.max(filtered_abs) > 0:
                        final_array = (255.0 * filtered_abs / np.max(filtered_abs)).astype(np.uint8)
                    else:
                        final_array = np.zeros_like(filtered, dtype=np.uint8)
                else:
                    # Для фільтрів розмиття просто обрізаємо значення
                    final_array = np.clip(filtered, 0, 255).astype(np.uint8)

                self.current_image = Image.fromarray(final_array)
                self.display_processed_image()
                self.update_image_info()

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
            self.update_filter_matrix_display(filter_name, kernel)
            self.status_bar.config(text=f"Застосовано фільтр: {filter_name}")
    
    def apply_prewitt(self):
        """Застосування фільтра Прюіта"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
        try:
            # 1. Надійно конвертуємо зображення у напівтонове ('L' - Luminance)
            grayscale_image = self.current_image.convert('L')
            # 2. Створюємо масив NumPy вже з гарантовано правильного формату
            img_array = np.array(grayscale_image, dtype=np.float64)
            
            # Застосування фільтрів Прюіта
            prewitt_x = ndimage.convolve(img_array, self.predefined_filters['Prewitt X'])
            prewitt_y = ndimage.convolve(img_array, self.predefined_filters['Prewitt Y'])
            
            # Обчислення градієнта (величина вектора градієнта)
            gradient = np.sqrt(prewitt_x**2 + prewitt_y**2)
            
            # Нормалізація до діапазону 0-255
            if np.max(gradient) > 0:
                gradient = (255.0 * gradient / np.max(gradient)).astype(np.uint8)
            else:
                gradient = np.zeros_like(gradient, dtype=np.uint8)
            
            self.current_image = Image.fromarray(gradient)
            self.display_processed_image()
            self.update_image_info()
            self.update_filter_matrix_display("Prewitt", [self.predefined_filters['Prewitt X'], self.predefined_filters['Prewitt Y']])
            self.status_bar.config(text="Застосовано фільтр Прюіта")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра Прюіта:\n{str(e)}")


    def apply_sobel(self):
        """Застосування фільтра Собеля"""
        if not self.current_image:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення")
            return
        try:
            # 1. Надійно конвертуємо зображення у напівтонове ('L' - Luminance)
            grayscale_image = self.current_image.convert('L')
            # 2. Створюємо масив NumPy вже з гарантовано правильного формату
            img_array = np.array(grayscale_image, dtype=np.float64)
            
            # Застосування фільтрів Собеля
            sobel_x = ndimage.convolve(img_array, self.predefined_filters['Sobel X'])
            sobel_y = ndimage.convolve(img_array, self.predefined_filters['Sobel Y'])
            
            # Обчислення градієнта (величина вектора градієнта)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Нормалізація до діапазону 0-255
            if np.max(sobel) > 0:
                sobel = (255.0 * sobel / np.max(sobel)).astype(np.uint8)
            else:
                sobel = np.zeros_like(sobel, dtype=np.uint8)
            
            self.current_image = Image.fromarray(sobel)
            self.display_processed_image()
            self.update_image_info()
            self.update_filter_matrix_display("Sobel", [self.predefined_filters['Sobel X'], self.predefined_filters['Sobel Y']])
            self.status_bar.config(text="Застосовано фільтр Собеля")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при застосуванні фільтра Собеля:\n{str(e)}")
    
    def open_image(self):
        """Відкриття зображення"""
        file_path = filedialog.askopenfilename(
        title="Виберіть зображення",
        filetypes=[
            ("JPEG Image", "*.jpeg"),
            ("JPG Image", "*.jpg"),
            ("PNG Image", "*.png"),
            ("TIFF Image", "*.tiff"),
            ("TIF Image", "*.tif"),
            ("BMP Image", "*.bmp"),
            ("GIF Image", "*.gif"),
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
                self.display_original_image()
                self.display_processed_image()                
                # Оновлення інформації
                self.update_image_info()
                self.status_bar.config(text=f"Завантажено: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося відкрити зображення:\n{str(e)}")


    def _display_image_on_canvas(self, image_to_display, canvas, photo_image_attr):
        """Відображає задане зображення на заданому Canvas."""
        if not image_to_display:
            return

        # Отримання розмірів canvas після того, як вікно буде промальовано
        self.root.update_idletasks() 
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Якщо canvas ще не має розміру, використовуємо тимчасовий
            canvas_width, canvas_height = 600, 600

        img_width, img_height = image_to_display.size
        
        # Масштабування для вписування в canvas
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.98 # 0.98 для невеликих відступів
        
        if scale < 1.0: # Зменшуємо тільки якщо зображення більше за canvas
             new_width = int(img_width * scale)
             new_height = int(img_height * scale)
             display_image = image_to_display.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            display_image = image_to_display
       
        # Конвертація в PhotoImage
        photo_image = ImageTk.PhotoImage(display_image)
        setattr(self, photo_image_attr, photo_image) # Зберігаємо посилання!

        # Очищення canvas і відображення зображення
        canvas.delete("all")
        canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=photo_image)

    # --- НОВІ МЕТОДИ: Окремо для кожної панелі ---
    def display_original_image(self):
        """Відображає оригінальне зображення в лівій панелі."""
        self._display_image_on_canvas(self.original_image, self.original_canvas, 'original_photo_image')


    def display_processed_image(self):
        """Відображає поточне (оброблене) зображення в правій панелі."""
        self._display_image_on_canvas(self.current_image, self.processed_canvas, 'processed_photo_image')
    
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
                self.display_processed_image()
                
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
            self.display_processed_image()
            self.update_image_info()
            self.update_filter_matrix_display("Оригінал")
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
    
    def update_filter_matrix_display(self, filter_name, kernel=None):
        """Оновлення відображення матриці фільтра"""
        # Очищення попередньої матриці
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        
        if kernel is not None:
            # Оновлення заголовка
            self.matrix_title_label.config(text=f"Фільтр: {filter_name}")
            
            # Спеціальна обробка для Prewitt та Sobel (два ядра)
            if isinstance(kernel, list) and len(kernel) == 2:
                # X ядро
                x_label = ttk.Label(self.matrix_frame, text="X:", font=('Arial', 9, 'bold'))
                x_label.grid(row=0, column=0, padx=5, pady=2, sticky='w')
                
                x_matrix_frame = ttk.Frame(self.matrix_frame)
                x_matrix_frame.grid(row=0, column=1, padx=5, pady=2)
                
                for i, row in enumerate(kernel[0]):
                    for j, val in enumerate(row):
                        entry = ttk.Entry(x_matrix_frame, width=8)
                        entry.grid(row=i, column=j, padx=1, pady=1)
                        entry.insert(0, f"{val:.2f}")
                        entry.config(state='readonly')
                
                # Y ядро
                y_label = ttk.Label(self.matrix_frame, text="Y:", font=('Arial', 9, 'bold'))
                y_label.grid(row=1, column=0, padx=5, pady=2, sticky='w')
                
                y_matrix_frame = ttk.Frame(self.matrix_frame)
                y_matrix_frame.grid(row=1, column=1, padx=5, pady=2)
                
                for i, row in enumerate(kernel[1]):
                    for j, val in enumerate(row):
                        entry = ttk.Entry(y_matrix_frame, width=8)
                        entry.grid(row=i, column=j, padx=1, pady=1)
                        entry.insert(0, f"{val:.2f}")
                        entry.config(state='readonly')
            else:
                # Звичайне ядро
                for i, row in enumerate(kernel):
                    for j, val in enumerate(row):
                        entry = ttk.Entry(self.matrix_frame, width=8)
                        entry.grid(row=i, column=j, padx=1, pady=1)
                        entry.insert(0, f"{val:.2f}")
                        entry.config(state='readonly')
        else:
            # Очищення заголовка
            self.matrix_title_label.config(text="Фільтр не застосовано")
    
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
            self.display_processed_image()
            self.update_image_info()
            self.update_filter_matrix_display("Gaussian HighPass", gaussian_kernel)
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
            self.display_processed_image()
            self.update_image_info()
            self.update_filter_matrix_display("Laplacian LowPass", lowpass_kernel)
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
