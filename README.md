# 🔐 Streamlit + Auth0 Login App

Ứng dụng demo đăng nhập bằng Auth0.

## ✅ Hướng dẫn triển khai trên Streamlit Cloud

1. Tạo project trên [Auth0](https://auth0.com/), thêm `Allowed Callback URL`:
   ```
   https://your-streamlit-username-auth0-streamlit-app.streamlit.app/
   ```

2. Vào Streamlit Cloud > Add Secrets và dán nội dung từ `.streamlit/secrets.toml`

3. Deploy repo, là xong 🎉

## 🛠️ Dependencies

```
pip install -r requirements.txt
```