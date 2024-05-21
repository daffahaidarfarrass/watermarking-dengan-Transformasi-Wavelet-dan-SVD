import numpy as np
import pywt
from PIL import Image
from PIL import ImageOps
import scipy.linalg
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("green")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Aplikasi Watermarking")
        self.geometry(f"{1200}x650")

        # create labels, buttons, and entry widgets
        self.image_label = customtkinter.CTkLabel(self, text="Pilih gambar:")
        self.image_label.pack(pady=10)

        self.upload_button = customtkinter.CTkButton(self, text="Unggah Gambar", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.watermark_label = customtkinter.CTkLabel(self, text="Teks watermark:")
        self.watermark_label.pack(pady=10)

        self.watermark_entry = customtkinter.CTkEntry(self)
        self.watermark_entry.pack(pady=10)

        self.alpha_label = customtkinter.CTkLabel(self, text="Skala alpha:")
        self.alpha_label.pack(pady=10)

        self.alpha_entry = customtkinter.CTkEntry(self)
        self.alpha_entry.pack(pady=10)

        self.submit_button = customtkinter.CTkButton(self, text="Proses", command=self.process_image)
        self.submit_button.pack(pady=10)

        self.compare_label = customtkinter.CTkLabel(self, text="Pilih gambar untuk dibandingkan:")
        self.compare_label.pack(pady=10)

        self.compare_button = customtkinter.CTkButton(self, text="Unggah Gambar", command=self.upload_compare_image)
        self.compare_button.pack(pady=10)

        self.compare_button = customtkinter.CTkButton(self, text="Bandingkan", command=self.compare_images)
        self.compare_button.pack(pady=10)

        # atribut untuk menyimpan path gambar yang diunggah
        self.path_to_image = None
        self.compare_path_to_image = None

    # Fungsi untuk menerapkan DWT pada gambar
    def apply_dwt(self, image_array):
        coeffs = pywt.dwt2(image_array, 'haar')
        cA, (cH, cV, cD) = coeffs
        return coeffs, cA, cH, cV, cD

    # Fungsi untuk menerapkan inverse DWT pada gambar
    def apply_idwt(self, coeffs):
        return pywt.idwt2(coeffs, 'haar')

    # Fungsi untuk menyisipkan watermark menggunakan SVD
    def embed_watermark_svd(self, cA, watermark, alpha):
        U, s, Vh = scipy.linalg.svd(cA, full_matrices=False)
        # Ubah watermark menjadi vektor dengan ukuran yang sama dengan s
        watermark_resized = np.resize(watermark, s.shape)
        s_w = s + alpha * watermark_resized
        return U, s_w, Vh

    # Fungsi untuk mengekstrak watermark menggunakan SVD
    def extract_watermark_svd(self, U, s_w, Vh, original_s, alpha):
        return (s_w - original_s) / alpha

    # Fungsi untuk memilih gambar dari sistem
    def upload_image(self):
        self.path_to_image = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
        if self.path_to_image:
            self.image = Image.open(self.path_to_image)
            messagebox.showinfo("Info", "Gambar berhasil diupload")

    # Fungsi untuk memilih gambar pembanding dari sistem
    def upload_compare_image(self):
        self.compare_path_to_image = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
        if self.compare_path_to_image:
            self.compare_image = Image.open(self.compare_path_to_image)
            messagebox.showinfo("Info", "Gambar pembanding berhasil diupload")

    # Fungsi untuk memproses gambar
    def process_image(self):
        if not self.path_to_image:
            messagebox.showerror("Error", "Mohon unggah gambar terlebih dahulu")
            return

        # Mendapatkan input dari pengguna
        watermark_text = self.watermark_entry.get()
        alpha = float(self.alpha_entry.get())

        # Memuat gambar tanpa mengonversinya ke grayscale
        image = Image.open(self.path_to_image)
        image_array = np.array(image)

        # Pisahkan saluran warna
        red_channel, green_channel, blue_channel = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]

        # Menerapkan DWT dan SVD pada setiap saluran
        coeffs_r, cA_r, cH_r, cV_r, cD_r = self.apply_dwt(red_channel)
        coeffs_g, cA_g, cH_g, cV_g, cD_g = self.apply_dwt(green_channel)
        coeffs_b, cA_b, cH_b, cV_b, cD_b = self.apply_dwt(blue_channel)

        # Mengonversi teks watermark ke array biner
        watermark = np.array([ord(char) for char in watermark_text]) > 0

        # Menyisipkan watermark menggunakan SVD pada setiap saluran
        U_r, s_w_r, Vh_r = self.embed_watermark_svd(cA_r, watermark, alpha)
        U_g, s_w_g, Vh_g = self.embed_watermark_svd(cA_g, watermark, alpha)
        U_b, s_w_b, Vh_b = self.embed_watermark_svd(cA_b, watermark, alpha)

        # Menerapkan inverse DWT untuk mendapatkan gambar dengan watermark pada setiap saluran
        coeffs_watermarked_r = (U_r @ np.diag(s_w_r) @ Vh_r, (cH_r, cV_r, cD_r))
        coeffs_watermarked_g = (U_g @ np.diag(s_w_g) @ Vh_g, (cH_g, cV_g, cD_g))
        coeffs_watermarked_b = (U_b @ np.diag(s_w_b) @ Vh_b, (cH_b, cV_b, cD_b))

        watermarked_image_array_r = self.apply_idwt(coeffs_watermarked_r)
        watermarked_image_array_g = self.apply_idwt(coeffs_watermarked_g)
        watermarked_image_array_b = self.apply_idwt(coeffs_watermarked_b)

        # Gabungkan kembali saluran warna
        watermarked_image_array = np.stack((watermarked_image_array_r, watermarked_image_array_g, watermarked_image_array_b), axis=-1)

        # Pastikan nilai pixel berada dalam rentang yang valid
        watermarked_image_array = np.clip(watermarked_image_array, 0, 255)

        self.watermarked_image = Image.fromarray(np.uint8(watermarked_image_array))

        # Menyimpan gambar dengan watermark
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            self.watermarked_image.save(save_path)
            messagebox.showinfo("Info", f"Gambar dengan watermark telah disimpan di {save_path}")


    # Fungsi untuk membandingkan gambar
    def compare_images(self):
        if not self.path_to_image or not self.compare_path_to_image:
            messagebox.showerror("Error", "Mohon unggah kedua gambar terlebih dahulu")
            return

        # Mengambil data gambar utama
        image = Image.open(self.path_to_image).convert('L')
        image_array = np.array(image)

        # Mengambil data gambar pembanding
        compare_image = Image.open(self.compare_path_to_image).convert('L')

        # Mengubah ukuran gambar pembanding agar sesuai dengan gambar utama
        compare_image = ImageOps.fit(compare_image, image.size, Image.Resampling.LANCZOS)
        compare_image_array = np.array(compare_image)

        # Menghitung perbedaan antara dua gambar
        difference = np.sum(np.abs(image_array - compare_image_array))

        if difference > (len(image_array)*1/100):
            messagebox.showinfo("Info", f"Gambar tidak asli\nPerbedaan antara dua gambar adalah: {difference} pixel")
        else:
            messagebox.showinfo("Info", f"Gambar asli\nPerbedaan antara dua gambar adalah: {difference} pixel")


if __name__ == "__main__":
    app = App()
    app.mainloop()
