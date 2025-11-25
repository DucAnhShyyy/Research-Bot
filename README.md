## Hướng dẫn chạy code cho anh em này:

### Bước 1: 
Clone Repo này về 1 thư mục của máy tính cá nhân
```
git clone https://github.com/DucAnhShyyy/Research-Bot.git
```

### Bước 2: Tạo môi trường ảo
```
python -m venv {tên môi trường ảo tự đặt}
{tên môi trường ảo tự đặt}\Scripts\activate
```

### Bước 3: Cấu hình
* Xem file notes.txt các dòng đầu để cấu hình lại vào file .env (Tự tạo nhé)
* Mở Docker Desktop lên trước
* Vào Terminal VS Code
```
docker compose up -d
```
* Check trong Docker Desktop xem image đã hoạt động chưa, các lần sau vào lại chỉ cần
```
docker start {tên image}\
```
* Tải thư viện:
```
pip install -r dependencies.txt
```

### Bước 4:
* Chạy web server
```
python src.app_gradio
```
