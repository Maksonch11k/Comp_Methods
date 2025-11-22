import numpy as np
import scipy.linalg
from PIL import Image, ImageDraw
import struct
import os
import math

class SVDCompressedImage:
    MAGIC = b'SVDZ'
    HEADER_FORMAT = '<4sIII' 
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def __init__(self, h=0, w=0, k=0, u=None, s=None, vt=None):
        self.h = h
        self.w = w
        self.k = k
        self.u = u
        self.s = s
        self.vt = vt

    def save(self, path):
        with open(path, 'wb') as f:
            header = struct.pack(self.HEADER_FORMAT, self.MAGIC, self.h, self.w, self.k)
            f.write(header)
            f.write(self.u.astype(np.float32).tobytes())
            f.write(self.s.astype(np.float32).tobytes())
            f.write(self.vt.astype(np.float32).tobytes())

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = f.read(cls.HEADER_SIZE)
            magic, h, w, k = struct.unpack(cls.HEADER_FORMAT, data)
            
            if magic != cls.MAGIC:
                raise ValueError("Incorrect file format")

            size_u = h * k * 4
            size_s = k * 4
            size_vt = k * w * 4
            
            u = np.frombuffer(f.read(size_u), dtype=np.float32).reshape((h, k))
            s = np.frombuffer(f.read(size_s), dtype=np.float32)
            vt = np.frombuffer(f.read(size_vt), dtype=np.float32).reshape((k, w))
            
            return cls(h, w, k, u, s, vt)

    def to_matrix(self):
        return (self.u * self.s) @ self.vt

def compress_image(input_path, output_path, N):
    img = Image.open(input_path).convert('L')
    matrix = np.array(img)
    h, w = matrix.shape
    
    original_size = os.path.getsize(input_path)
    target_size = original_size // N
    
    available_bytes = target_size - SVDCompressedImage.HEADER_SIZE
    bytes_per_rank = 4 * (h + w + 1)
    
    k = int(available_bytes // bytes_per_rank)
    k = max(1, min(k, min(h, w)))
    
    print(f"  Compressing N={N}: Original={original_size}B to Target={target_size}B. Rank k={k}")
    
    U, S, Vt = np.linalg.svd(matrix.astype(float), full_matrices=False)
    
    svd_obj = SVDCompressedImage(h, w, k, U[:, :k], S[:k], Vt[:k, :])
    svd_obj.save(output_path)
    
    real_size = os.path.getsize(output_path)
    print(f"  Saved: {real_size} B")

def decompress_image(input_path, output_path):
    svd_obj = SVDCompressedImage.load(input_path)
    matrix = svd_obj.to_matrix()
    matrix = np.clip(matrix, 0, 255).astype(np.uint8)
    img = Image.fromarray(matrix)
    img.save(output_path)

def bad_matrix(m, n):
    i = np.arange(m).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    A = (i * j) % 256
    return A.astype(float)

def calculate_difference_metric(s1, s2):
    a = np.sort(s1)
    b = np.sort(s2)
    c = np.zeros_like(a)
    
    for j in range(len(a)):
        if a[j] == 0 or b[j] == 0:
            c[j] = 0
        else:
            c[j] = max(a[j] / b[j], b[j] / a[j])
            
    return np.linalg.norm(c)

def part2_comparison():
    print("\n=== Part 2 ===")

    h, w = 533, 800
    i = np.arange(800).reshape(-1, 1)
    j = np.arange(533).reshape(1, -1)
    A = (i * j) % 256
    A.astype(float)
    
    Image.fromarray(np.clip(A, 0, 255).astype(np.uint8), mode='L').save("part2_bad_matrix.bmp")
    img_loaded = Image.open("part2_bad_matrix.bmp").convert('L')
    A_loaded = np.array(img_loaded, dtype=np.float64)
    U1, s1, Vt1 = np.linalg.svd(A_loaded)

    try:
        U2, s2, Vt2 = scipy.linalg.svd(A_loaded, lapack_driver='gesvd')
    except:
        U2, s2, Vt2 = scipy.linalg.svd(A_loaded)

    metric = calculate_difference_metric(s1, s2)
    
    print(f"\nResult Metric L2: {metric}")

def generate_examples():
    os.makedirs("output", exist_ok=True)
    
    img1 = np.fromfunction(lambda i, j: (i+j)//4, (512, 512)).astype(np.uint8)
    Image.fromarray(img1).save("output/ex1_gradient.bmp")
    
    img2_pil = Image.new('L', (512, 512), 0)
    draw = ImageDraw.Draw(img2_pil)
    draw.ellipse((100, 100, 412, 412), fill=255)
    img2_pil.save("output/ex2_circle.bmp")
    
    img3 = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    Image.fromarray(img3).save("output/ex3_noise.bmp")
    
    return ["output/ex1_gradient.bmp", "output/ex2_circle.bmp", "output/ex3_noise.bmp"]

if __name__ == "__main__":
    input_files = generate_examples()
    
    print("\n=== Part 1: ===")
    factors = [2, 4, 8]
    
    for fpath in input_files:
        fname = os.path.basename(fpath)
        print(f"\n{fname}")
        for N in factors:
            base_name = fname.replace('.bmp', '')
            compressed = f"output/{base_name}_N{N}.svdz"
            restored = f"output/{base_name}_N{N}_restored.bmp"
            
            compress_image(fpath, compressed, N)
            decompress_image(compressed, restored)
            
    part2_comparison()