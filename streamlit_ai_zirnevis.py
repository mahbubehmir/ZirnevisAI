import streamlit as st
import subprocess
import librosa
import noisereduce as nr
import soundfile as sf
import os
import whisper
import json
from deep_translator import GoogleTranslator

# تنظیمات پیشرفته
TEMP_FILES = [
    "input_video.mp4", "extracted_audio.mp3",
    "cleaned_audio.wav", "final_audio.wav",
    "transcribed_text_whisper.json", "transcribed_text_whisper.txt",
    "subtitles_editable.txt", "edited_subtitles.txt",
    "final_subtitles.srt", "output_with_subtitles.mp4"
]

class Config:
    MODEL_SIZE = "small"
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 50MB

config = Config()

def cleanup_files():
    """پاکسازی فایل های موقت"""
    for file in TEMP_FILES:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            st.error(f"خطا در پاکسازی {file}: {str(e)}")

def format_time(seconds: float) -> str:
    """فرمت دهی زمان برای زیرنویس"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def create_editable_txt(transcription_result: dict) -> None:
    """ایجاد فایل متنی قابل ویرایش با فرمت ساده"""
    with open("subtitles_editable.txt", "w", encoding="utf-8") as f:
        for seg in transcription_result["segments"]:
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            text = seg["text"].replace("\n", " ")
            translated = GoogleTranslator(source='auto', target='fa').translate(text)
            
            f.write(f"{start} | {end} | {text} | {translated}\n")

def txt_to_srt(input_txt: str, output_srt: str) -> None:
    """تبدیل فایل متنی ویرایش شده به فرمت SRT"""
    with open(input_txt, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    
    with open(output_srt, "w", encoding="utf-8") as fout:
        for i, line in enumerate(lines):
            parts = line.strip().split("|")
            if len(parts) != 4:
                raise ValueError("فرمت فایل نامعتبر است")
            
            start, end, text, translated = [part.strip() for part in parts]
            
            fout.write(f"{i+1}\n")
            fout.write(f"{start} --> {end}\n")
            fout.write(f"{text}\n{translated}\n\n")

# تنظیمات صفحه
st.set_page_config(page_title="تولید زیرنویس هوشمند", layout="wide")
st.title("سیستم تولید زیرنویس دوزبانه با هوش مصنوعی")

# مراحل پردازش
st.header("مراحل کار:")
st.markdown("""
1. ویدیو را آپلود کنید (حداکثر 500MB)
2. فایل زیرنویس خودکار را دانلود و ویرایش کنید
3. فایل ویرایش شده را آپلود کنید
4. ویدیوی نهایی با زیرنویس را دریافت کنید
""")

# آپلود ویدیو
video_file = st.file_uploader("مرحله 1: ویدیو را انتخاب کنید", type=["mp4", "mov", "avi"])

if video_file:
    with st.spinner("در حال پردازش ویدیو..."):
        try:
            cleanup_files()
            
            # ذخیره ویدیو
            with open("input_video.mp4", "wb") as f:
                f.write(video_file.getbuffer())
            
            # بررسی حجم
            if os.path.getsize("input_video.mp4") > config.MAX_VIDEO_SIZE:
                st.error("حجم ویدیو باید کمتر از 500 مگابایت باشد")
                cleanup_files()
                st.stop()

            # استخراج صدا
            subprocess.run([
                'ffmpeg', '-i', 'input_video.mp4',
                '-q:a', '0', '-map', 'a', 'extracted_audio.mp3'
            ], check=True)
            
            # کاهش نویز
            y, sr = librosa.load("extracted_audio.mp3", sr=None)
            reduced_audio = nr.reduce_noise(y=y, sr=sr)
            sf.write("cleaned_audio.wav", reduced_audio, sr)
            
            # تشخیص گفتار
            model = whisper.load_model(config.MODEL_SIZE, device="cpu")
            result = model.transcribe("cleaned_audio.wav")
            
            # ایجاد فایل متنی
            create_editable_txt(result)
            
            st.success("زیرنویس خودکار آماده شد!")
            st.download_button(
                label="دانلود فایل زیرنویس برای ویرایش",
                data=open("subtitles_editable.txt", "rb"),
                file_name="subtitles_editable.txt",
                mime="text/plain"
            )
            
            # آپلود فایل ویرایش شده
            edited_subtitles = st.file_uploader("مرحله 2: فایل ویرایش شده را آپلود کنید", type="txt")
            
            if edited_subtitles:
                with st.spinner("در حال تولید ویدیو نهایی..."):
                    with open("edited_subtitles.txt", "wb") as f:
                        f.write(edited_subtitles.getbuffer())
                    
                    # تبدیل به SRT
                    txt_to_srt("edited_subtitles.txt", "final_subtitles.srt")
                    
                    # اضافه کردن زیرنویس
                    subprocess.run([
                        'ffmpeg', '-i', 'input_video.mp4',
                        '-vf', "subtitles=final_subtitles.srt:force_style='FontName=Arial,Fontsize=24,PrimaryColour=&H00FFFFFF,Outline=1'",
                        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
                        '-c:a', 'copy', 'output_with_subtitles.mp4'
                    ], check=True)
                    
                    st.success("ویدیوی نهایی آماده شد!")
                    st.video("output_with_subtitles.mp4")
                    st.download_button(
                        label="دانلود ویدیو با زیرنویس",
                        data=open("output_with_subtitles.mp4", "rb"),
                        file_name="output_with_subtitles.mp4",
                        mime="video/mp4"
                    )
        
        except Exception as e:
            st.error(f"خطا در پردازش: {str(e)}")
            cleanup_files()