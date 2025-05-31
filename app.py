import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ชื่อแอป
st.set_page_config(page_title="XGBoost Loan Prediction", layout="wide")
st.title("📊 Loan Approval Prediction using XGBoost")

# อัปโหลดไฟล์
uploaded_file = st.file_uploader("📁 อัปโหลดไฟล์ CSV ของคุณที่มีคอลัมน์ 'status'", type=["csv"])

if uploaded_file is not None:
    try:
        # อ่านไฟล์ CSV
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()  # ลบช่องว่างจากชื่อคอลัมน์

        st.subheader("🔍 ข้อมูลเบื้องต้น")
        st.dataframe(df.head())

        if 'status' not in df.columns:
            st.error("❌ ไม่พบคอลัมน์ 'status' ในไฟล์ CSV ของคุณ กรุณาตรวจสอบอีกครั้ง")
        else:
            # แยก Features และ Target
            X = df.drop('status', axis=1)
            y = df['status']

            # เข้ารหัส Target
            le = LabelEncoder()
            y = le.fit_transform(y)

            # เข้ารหัส Features (เฉพาะ categorical)
            X = pd.get_dummies(X)

            # แบ่งข้อมูล train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # สร้างและเทรนโมเดล XGBoost
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)

            # ทำนายผล
            y_pred = model.predict(X_test)

            # แสดงผลลัพธ์
            st.subheader("✅ Accuracy")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")

            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            st.write(pd.DataFrame(cm, 
                                  columns=[f"Predicted {cls}" for cls in le.classes_],
                                  index=[f"Actual {cls}" for cls in le.classes_]))

            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
else:
    st.info("⬆️ กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการวิเคราะห์")
