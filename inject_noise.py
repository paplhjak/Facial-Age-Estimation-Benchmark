import random
import argparse
import numpy as np
import pandas as pd

def apply_noise_to_dataset(csv_path, npy_path, seed=None):
    # تنظیم seed برای بازتولید نتایج
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # خواندن فایل CSV
    df = pd.read_csv(csv_path, header=None)
    
    # خواندن ماتریس انتقال نویز (noise transition matrix)
    noise_matrix = np.load(npy_path)
    
    # شناسایی داده‌های آموزشی (train) که ستون سوم آن‌ها برابر با 0 است
    train_data = df[df[2] < 2]
    
    # شناسایی کلاس‌های متمایز
    unique_classes = train_data[3].unique()
    
    # ذخیره کردن تغییرات در یک لیست جدید
    noisy_data = df.copy()
    
    # بررسی برای هر کلاس منحصر به فرد
    for class_label in unique_classes:
        # داده‌های کلاس خاص
        class_data = train_data[train_data[3] == class_label]
        num_class_samples = len(class_data)
        
        # ماتریس انتقال نویز مربوط به این کلاس
        noise_row = noise_matrix[class_label]
        
        # تعداد داده‌های نویزی که باید به هر کلاس اختصاص یابد
        noisy_samples_count = (noise_row * num_class_samples).astype(int)
        
        # انتخاب داده‌ها به صورت تصادفی و تغییر لیبل آنها
        already_selected = set()  # مجموعه‌ای برای جلوگیری از انتخاب مجدد داده‌ها
        for i, noisy_class in enumerate(noisy_samples_count):
            for _ in range(noisy_class):
                # انتخاب تصادفی یک نمونه از این کلاس
                while True:
                    random_index = random.choice(class_data.index)
                    if random_index not in already_selected:
                        already_selected.add(random_index)
                        break
                # تغییر لیبل داده به کلاس نویزی
                noisy_data.loc[random_index, 3] = i
    
    # ذخیره کردن دیتاست نویزی شده با نام جدید
    # noisy_csv_path = csv_path.replace(".csv", "_noisy.csv")
    noisy_data.to_csv(csv_path, header=False, index=False)

    print(f"Dataset with noise saved as: {csv_path}")
    
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Inject noise into training data based on a noise transition matrix.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('npy_file', type=str, help='Path to the noise transition matrix (npy file).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default is 42).')
    
    args = parser.parse_args()
    
    apply_noise_to_dataset(args.csv_file, args.npy_file, seed=args.seed)

if __name__ == '__main__':
    main()
