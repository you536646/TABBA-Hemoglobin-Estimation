# استخدام بيئة بايثون 3.10 متوافقة مع XGBoost و Linux
FROM python:3.10-slim

# تعيين مسار العمل داخل السيرفر
WORKDIR /app

# تثبيت متطلبات النظام الأساسية (مهمة جداً لـ OpenCV و YOLO)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المكتبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كل ملفات المشروع (الموديل، الأكواد، الواجهة)
COPY . .

# أمر تشغيل السيرفر (Render سيقوم بتوفير البورت أوتوماتيكياً)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
