import cv2
import numpy as np
import cmath
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import math

# --- Функції прямого та оберненого перетворення Фур'є ---

def dft_1d(signal, inverse=False):
    """Самописне 1D ДПФ та Обернене ДПФ"""
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    sign = 1j if inverse else -1j
    for k in range(N):
        s = 0j
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            s += signal[n] * cmath.exp(sign * angle)
        result[k] = s
    if inverse:
        result /= N
    return result

def fft2d(image, inverse=False):
    """2D ДПФ/ОДПФ через послідовні 1D перетворення"""
    M, N = image.shape
    temp = np.zeros((M, N), dtype=complex)
    for i in range(M):
        temp[i, :] = dft_1d(image[i, :], inverse)
    
    result = np.zeros((M, N), dtype=complex)
    for j in range(N):
        result[:, j] = dft_1d(temp[:, j], inverse)
        
    return result

def shift_image(img):
    """Множимо на (-1)^(x+y) для центру спектра"""
    M, N = img.shape
    img = img.astype(float)
    shifted = np.zeros_like(img, dtype=float)
    for x in range(M):
        for y in range(N):
            shifted[x, y] = img[x, y] * ((-1) ** (x + y))
    return shifted


# --- Функції для створення фільтрів ---

def create_filter_mask(shape, D0, n, filter_type, pass_type):
    """Створює маску фільтра на основі заданих параметрів"""
    M, N = shape
    H = np.zeros((M, N), dtype=np.float32)
    center = (M // 2, N // 2)

    for u in range(M):
        for v in range(N):
            dist = math.sqrt((u - center[0])**2 + (v - center[1])**2)
            
            if filter_type == 'ideal':
                H[u, v] = 1 if dist <= D0 else 0
            elif filter_type == 'gaussian':
                if D0 > 0:
                    H[u, v] = math.exp(-(dist**2) / (2 * D0**2))
            elif filter_type == 'butterworth':
                if D0 > 0:
                    H[u, v] = 1 / (1 + (dist / D0)**(2 * n))
    
    if pass_type == 'high':
        H = 1 - H
        
    return H

# --- Клас GUI ---

class FourierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аналізатор та фільтр Фур'є-спектру")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2e2e2e")

        self.f_transform = None # Для зберігання результату ДПФ
        self.original_image_small = None

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#2e2e2e")
        style.configure("TLabel", background="#2e2e2e", foreground="white", font=("Arial", 11))
        style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        style.configure("TButton", font=("Arial", 12), background="#007bff", foreground="white", padding=10)
        style.map("TButton", background=[('active', '#0056b3'), ('disabled', '#555555')])
        style.configure("TLabelframe", background="#3c3c3c", bordercolor="#555")
        style.configure("TLabelframe.Label", background="#3c3c3c", foreground="white", font=("Arial", 12, "bold"))
        style.configure("TCombobox", fieldbackground="#555", background="#555", foreground="white")

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Верхня панель ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=10)
        ttk.Label(top_frame, text="Фільтрація зображень в частотній області", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Відкрити зображення", command=self.open_image).pack(side=tk.RIGHT)

        # --- Панель керування фільтрами ---
        controls_frame = ttk.LabelFrame(main_frame, text="Параметри фільтрації", padding="10")
        controls_frame.pack(fill=tk.X, pady=10)
        self.create_filter_controls(controls_frame)

        # --- Область для зображень (2x3) ---
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(expand=True, fill=tk.BOTH, pady=10)
        for i in range(3): image_frame.columnconfigure(i, weight=1)
        for i in range(2): image_frame.rowconfigure(i, weight=1)

        self.img_labels = {}
        titles = ["Оригінал", "Амплітудний спектр", "Маска фільтра",
                  "Спектр після фільтрації", "Результат", ""] # Поки що пусте вікно
        
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for i, (title, pos) in enumerate(zip(titles, positions)):
            self.img_labels[i] = self.create_image_display(image_frame, pos[0], pos[1], title)
        
        self.img_labels[5].master.grid_forget() # Ховаємо зайвий елемент

    def create_filter_controls(self, parent):
        parent.columnconfigure((1, 3, 5), weight=1)

        # Тип фільтра (НЧ/ВЧ)
        ttk.Label(parent, text="Тип:").grid(row=0, column=0, padx=5, sticky="w")
        self.pass_type_var = tk.StringVar(value="Low-Pass")
        pass_type_cb = ttk.Combobox(parent, textvariable=self.pass_type_var, values=["Low-Pass", "High-Pass"], state="readonly")
        pass_type_cb.grid(row=0, column=1, padx=5, sticky="ew")
        pass_type_cb.bind("<<ComboboxSelected>>", self.apply_filter)

        # Вид фільтра
        ttk.Label(parent, text="Фільтр:").grid(row=0, column=2, padx=5, sticky="w")
        self.filter_type_var = tk.StringVar(value="Gaussian")
        filter_type_cb = ttk.Combobox(parent, textvariable=self.filter_type_var, values=["Ideal", "Gaussian", "Butterworth"], state="readonly")
        filter_type_cb.grid(row=0, column=3, padx=5, sticky="ew")
        filter_type_cb.bind("<<ComboboxSelected>>", self.apply_filter)

        # Частота зрізу D0
        ttk.Label(parent, text="Частота зрізу (D0):").grid(row=1, column=0, padx=5, sticky="w")
        self.d0_var = tk.IntVar(value=30)
        self.d0_scale = ttk.Scale(parent, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.d0_var, command=self.apply_filter)
        self.d0_scale.grid(row=1, column=1, columnspan=3, padx=5, sticky="ew")
        
        # Порядок n для Баттерворта
        self.n_label = ttk.Label(parent, text="Порядок (n):")
        self.n_label.grid(row=1, column=4, padx=5, sticky="w")
        self.n_var = tk.IntVar(value=2)
        self.n_scale = ttk.Scale(parent, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.n_var, command=self.apply_filter)
        self.n_scale.grid(row=1, column=5, padx=5, sticky="ew")

    def create_image_display(self, parent, row, col, title_text):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        
        ttk.Label(frame, text=title_text, font=("Arial", 12, "bold")).grid(row=0, column=0, pady=(0, 5))
        
        img_label = tk.Label(frame, bg="#404040", relief="solid", bd=1)
        img_label.grid(row=1, column=0, sticky="nsew")
        return img_label

    def open_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Зображення", "*.png *.jpg *.jpeg *.bmp *.tif")])
        if not filepath: return

        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None: raise ValueError("Не вдалося завантажити зображення")
            
            # Обробка
            self.original_image_small = cv2.resize(img, (256, 256))
            shifted = shift_image(self.original_image_small)
            self.f_transform = fft2d(shifted)
            
            # Відображення початкових результатів
            self.update_image_label(self.img_labels[0], self.original_image_small)
            
            magnitude_spectrum = 20 * np.log(np.abs(self.f_transform) + 1)
            self.update_image_label(self.img_labels[1], magnitude_spectrum)
            
            # Активуємо фільтрацію
            self.apply_filter()

        except Exception as e:
            messagebox.showerror("Помилка обробки", f"Сталася помилка: {e}")

    def apply_filter(self, event=None):
        if self.f_transform is None: return

        # Отримуємо параметри з GUI
        pass_type = self.pass_type_var.get().lower().split('-')[0]
        filter_type = self.filter_type_var.get().lower()
        d0 = self.d0_var.get()
        n = self.n_var.get()
        
        # Створюємо маску фільтра
        mask = create_filter_mask(self.f_transform.shape, d0, n, filter_type, pass_type)
        self.update_image_label(self.img_labels[2], mask * 255)

        # Застосовуємо фільтр
        filtered_f = self.f_transform * mask
        
        # Відображаємо відфільтрований спектр
        filtered_magnitude = 20 * np.log(np.abs(filtered_f) + 1)
        self.update_image_label(self.img_labels[3], filtered_magnitude)

        # Обернене перетворення
        img_back_shifted = fft2d(filtered_f, inverse=True)
        img_back = shift_image(np.real(img_back_shifted))
        
        # Відображаємо результат
        self.update_image_label(self.img_labels[4], img_back)

    def update_image_label(self, label, np_image):
        if np_image is None: return
        
        # Нормалізація для візуалізації
        norm_image = cv2.normalize(np_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        img = Image.fromarray(norm_image)
        
        self.root.update_idletasks()
        label_w, label_h = label.winfo_width(), label.winfo_height()
        
        if label_w > 1 and label_h > 1:
            img.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)
            
        img_tk = ImageTk.PhotoImage(image=img)
        label.config(image=img_tk)
        label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = FourierApp(root)
    root.mainloop()