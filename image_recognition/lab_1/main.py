import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageProcessor:
    def __init__(self, image_path):
        """
        Ініціалізація процесора зображень.
        
        Args:
            image_path (str): Шлях до зображення.
        """
        self.image_path = image_path
        self.original_image = None
        self.grayscale_image = None
        self.equalized_image = None

    def load_image(self):
        """Завантаження зображення."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Не вдалося завантажити зображення з {self.image_path}")
        print(f"Зображення завантажено. Розмір: {self.original_image.shape}")
        return self # Дозволяє створювати ланцюжки викликів

    def convert_to_grayscale(self):
        """Перетворення зображення в градації сірого."""
        if self.original_image is None:
            raise ValueError("Спочатку завантажте зображення")
        self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        print("Зображення перетворено в градації сірого.")
        return self

    def equalize_histogram(self):
        """Вирівнювання гістограми."""
        if self.grayscale_image is None:
            raise ValueError("Спочатку перетворіть зображення в градації сірого")
        self.equalized_image = cv2.equalizeHist(self.grayscale_image)
        print("Гістограма вирівняна.")
        return self

    def build_histogram(self, image):
        """Допоміжна функція для побудови гістограми."""
        return cv2.calcHist([image], [0], None, [256], [0, 256])


    def display_color_histogram(self, save_path=None):
        """Відображає гістограму для кожного BGR каналу."""
        if self.original_image is None:
            raise ValueError("Немає оригінального зображення для аналізу.")

        plt.figure(figsize=(8, 5))
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title("Гістограма кольорового зображення (BGR)")
        plt.xlim([0, 256])

        if save_path:
            plt.savefig(save_path)
            print(f"Графік збережено як {save_path}")
        plt.show()

    def display_grayscale_comparison(self, save_path=None):
        """Порівнює оригінальне сіре та вирівняне зображення."""
        if self.grayscale_image is None or self.equalized_image is None:
            raise ValueError("Необхідно спочатку обробити зображення.")
            
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(self.grayscale_image, cmap='gray')
        plt.title("Original Gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(self.equalized_image, cmap='gray')
        plt.title("Equalized Gray")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
            print(f"Графік збережено як {save_path}")
        plt.show()
    
    def display_histogram_cdf_comparison(self, save_path=None):
        """Порівнює гістограми та кумулятивні функції розподілу (CDF)."""
        if self.grayscale_image is None or self.equalized_image is None:
            raise ValueError("Необхідно спочатку обробити зображення.")

        # Розрахунки для оригінального сірого зображення
        hist_gray = self.build_histogram(self.grayscale_image)
        cdf_gray = hist_gray.cumsum()
        cdf_normalized = cdf_gray * hist_gray.max() / cdf_gray.max()

        # Розрахунки для вирівняного зображення
        hist_eq = self.build_histogram(self.equalized_image)
        cdf_eq = hist_eq.cumsum()
        cdf_eq_norm = cdf_eq * hist_eq.max() / cdf_eq.max()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(hist_gray, color='gray', label='Гістограма')
        plt.plot(cdf_normalized, color='red', label='CDF')
        plt.title("Гістограма + CDF (оригінал)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist_eq, color='black', label='Гістограма')
        plt.plot(cdf_eq_norm, color='red', label='CDF')
        plt.title("Гістограма + CDF (вирівняне)")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Графік збережено як {save_path}")
        plt.show()

    def run_full_analysis(self, output_dir="output"):
        """Виконує повний цикл обробки та візуалізації."""
        print("Початок повного аналізу зображення...")
        
        # Створюємо папку, якщо її немає
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Етапи обробки
        self.load_image()
        self.convert_to_grayscale()
        self.equalize_histogram()

        self.display_color_histogram(save_path=f"{output_dir}/hist_bgr.png")
        self.display_grayscale_comparison(save_path=f"{output_dir}/comparison_gray.png")
        self.display_histogram_cdf_comparison(save_path=f"{output_dir}/histograms_comparison.png")

        print("Аналіз завершено!")

if __name__ == "__main__":
    image_path = "D:\projects\labs\image_recognition\lab_1\photo_2025-09-27_12-53-02.jpg"
    
    try:
        processor = ImageProcessor(image_path)
        processor.run_full_analysis(output_dir="./lab1_output")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Помилка: {e}")
    except Exception as e:
        print(f"Сталася неочікувана помилка: {e}")