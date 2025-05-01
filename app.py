import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models.cnn_model import EyeCNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from fpdf import FPDF
import base64

# 🎓 Project Header
st.markdown("""
# 🧿 Eye Disease Detection & Classification  
### Govt. First Grade College for Women, Jamkhandi  
#### Dept. of Computer Science – BCA VI Sem Project
""")

# 🧠 Class Labels
class_names = ['Cataract', 'Glaucoma', 'Normal', 'Other']

# 🧠 Model Load
MODEL_PATH = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeCNN(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 🎯 Initialize Session State
if "all_preds" not in st.session_state:
    st.session_state.all_preds = []
    st.session_state.all_labels = []

# 🔄 Reset Session
if st.button("🔁 Reset Confusion Matrix Data"):
    st.session_state.all_preds.clear()
    st.session_state.all_labels.clear()
    st.success("Session data cleared. You can now start fresh.")

# 📂 Upload and Classify
st.markdown("### 📥 Upload Eye Images and Assign Labels")

uploaded_files = st.file_uploader("Upload one or more images...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Image: {uploaded_file.name}", use_container_width=True)

        true_label = st.selectbox(f"✅ Select actual disease for {uploaded_file.name}", class_names, key=uploaded_file.name)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        pred_label = class_names[pred.item()]
        st.success(f"🔍 Prediction: **{pred_label}** ({confidence.item()*100:.2f}%)")

        # Store results
        st.session_state.all_preds.append(pred.item())
        st.session_state.all_labels.append(class_names.index(true_label))

        # 🧾 Optional PDF Report
        if st.button(f"📄 Generate Report for {uploaded_file.name}"):
            class PDF(FPDF):
                def header(self):
                    self.set_font("Arial", "B", 12)
                    self.cell(0, 10, "Eye Disease Detection Report", ln=True, align="C")
                def body(self):
                    self.set_font("Arial", "", 12)
                    self.ln(10)
                    self.cell(0, 10, f"Filename: {uploaded_file.name}", ln=True)
                    self.cell(0, 10, f"Actual: {true_label}", ln=True)
                    self.cell(0, 10, f"Predicted: {pred_label}", ln=True)
                    self.cell(0, 10, f"Confidence: {confidence.item()*100:.2f}%", ln=True)

            pdf = PDF()
            pdf.add_page()
            pdf.body()
            pdf.output("report.pdf")
            with open("report.pdf", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Eye_Diagnosis_Report.pdf">📥 Download Report</a>', unsafe_allow_html=True)

# 📊 Confusion Matrix Display
if st.session_state.all_preds:
    st.markdown("### 📊 Confusion Matrix")
    cm = confusion_matrix(st.session_state.all_labels, st.session_state.all_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (All Predictions)")
    st.pyplot(fig)

# 👨‍🏫 Footer
st.markdown("---")
st.markdown("""
**Project Team**  
- Student: *[Your Name]*  
- Guide: *[Faculty Name]*  
- Semester: BCA VI  
- Academic Year: 2024–25  
""")
