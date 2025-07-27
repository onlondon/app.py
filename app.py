import streamlit as st
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px

# ===== Load model & sheet =====
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Trả về (model, feature_names)

@st.cache_data
def load_sheet(sheet_name="Sheet1"):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["google"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)
    client = gspread.authorize(creds)
    sheet = client.open("employee").worksheet(sheet_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# ===== Tiền xử lý =====
def preprocess(df, feature_names):
    df = df.copy()
    if 'Gender' in df.columns and df['Gender'].dtype == object:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    if 'OverTime' in df.columns and df['OverTime'].dtype == object:
        df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    return df[feature_names]

# ===== Streamlit layout =====
st.set_page_config(page_title="Dự đoán nghỉ việc", layout="wide")

# ===== Menu lựa chọn trang =====
page = st.sidebar.radio("📂 Chọn trang", ["Trang 1: Dự đoán & chỉnh sửa", "Trang 2: Phân tích dữ liệu"])

# ===== Load data + model =====
try:
    original_df = load_sheet()
    model = load_model()
    feature_names = model.feature_names_in_  # Nếu model hỗ trợ
    df_input = preprocess(original_df, feature_names)
    probas = model.predict_proba(df_input)[:, 1]
    original_df['Nguy cơ nghỉ việc (%)'] = (probas * 100).round(2)
    original_df['Dự đoán'] = ['Nghỉ việc' if p >= 0.5 else 'Ở lại' for p in probas]

    # ===== Trang 1 =====
    if page == "Trang 1: Dự đoán & chỉnh sửa":
        st.title("✏️ Chỉnh sửa dữ liệu nhân viên & Dự đoán nghỉ việc")
        editable_cols = ['BonusAmount', 'HourlyRate', 'JobSatisfaction', 'JobInvolvement']
        edited_df = st.data_editor(original_df, column_config={col: st.column_config.NumberColumn() for col in editable_cols}, disabled=[c for c in original_df.columns if c not in editable_cols], num_rows="dynamic")

        # Dự đoán lại sau khi sửa
        df_input_new = preprocess(edited_df, feature_names)
        new_probas = model.predict_proba(df_input_new)[:, 1]
        edited_df['Nguy cơ nghỉ việc (%)'] = (new_probas * 100).round(2)
        edited_df['Dự đoán'] = ['Nghỉ việc' if p >= 0.5 else 'Ở lại' for p in new_probas]

        st.subheader("📋 Kết quả dự đoán")
        st.dataframe(edited_df.style.background_gradient(cmap='OrRd', subset=['Nguy cơ nghỉ việc (%)']))

    # ===== Trang 2 =====
    else:
        st.title("📊 Phân tích dữ liệu nghỉ việc")

        st.subheader("🔝 Top 5 nhân viên có nguy cơ nghỉ việc cao nhất")
        top5 = original_df.sort_values(by='Nguy cơ nghỉ việc (%)', ascending=False).head(5)
        st.dataframe(top5.style.background_gradient(cmap='Reds', subset=['Nguy cơ nghỉ việc (%)']))

        st.subheader("📈 Biểu đồ tỉ lệ nghỉ việc")
        pie_data = original_df['Dự đoán'].value_counts().reset_index()
        pie_data.columns = ['Trạng thái', 'Số lượng']
        fig = px.pie(pie_data, names='Trạng thái', values='Số lượng',
                     color='Trạng thái', color_discrete_map={'Nghỉ việc': 'red', 'Ở lại': 'green'})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Trung bình nguy cơ nghỉ việc theo JobRole")
        if 'JobRole' in original_df.columns:
            role_avg = original_df.groupby("JobRole")["Nguy cơ nghỉ việc (%)"].mean().reset_index()
            fig2 = px.bar(role_avg, x="JobRole", y="Nguy cơ nghỉ việc (%)", color="Nguy cơ nghỉ việc (%)", color_continuous_scale="OrRd")
            st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"❌ Có lỗi xảy ra: {e}")
