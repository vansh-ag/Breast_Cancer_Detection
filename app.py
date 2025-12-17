import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)

# Model Architecture
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CNNTransformerHybrid(nn.Module):
    def __init__(self, num_classes=8, pretrained=False):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.cnn_layers = nn.Sequential(*list(resnet.children())[:-2])
        
        self.num_features = 2048
        self.num_patches = 49
        self.transformer_dim = 512
        self.num_heads = 8
        self.num_transformer_layers = 4
        
        self.projection = nn.Linear(self.num_features, self.transformer_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.transformer_dim,
                num_heads=self.num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1
            )
            for _ in range(self.num_transformer_layers)
        ])
        
        self.norm = nn.LayerNorm(self.transformer_dim)
        self.head = nn.Sequential(
            nn.Linear(self.transformer_dim, self.transformer_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.transformer_dim, num_classes)
        )
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.cnn_layers(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        x = self.projection(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformerHybrid(num_classes=8, pretrained=False)
    
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("❌ Model file 'best_model.pth' not found. Please ensure it's in the same directory as app.py")
        return None, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def create_probability_chart(probabilities, class_names):
    colors = ['#ef4444' if i < 4 else '#22c55e' for i in range(len(class_names))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities * 100,
            y=class_names,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p*100:.2f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Classification Probabilities by Cancer Type',
        xaxis_title='Confidence (%)',
        yaxis_title='Cancer Type',
        height=450,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def main():
    st.title("🎗️ Breast Cancer Detection System")
    st.markdown("""
    This AI-powered application uses a **CNN + Transformer Hybrid model** to analyze histopathology images 
    and classify different types of breast cancer with **95% accuracy**.
    
    **⚠️ Medical Disclaimer:** This is a research and educational tool. Always consult qualified healthcare 
    professionals for accurate medical diagnosis and treatment decisions.
    """)
    
    class_names = [
        'Papillary Carcinoma', 'Mucinous Carcinoma', 
        'Lobular Carcinoma', 'Ductal Carcinoma',
        'Tubular Adenoma', 'Phyllodes Tumor', 
        'Fibroadenoma', 'Adenosis'
    ]
    
    malignant_types = class_names[:4]
    benign_types = class_names[4:]
    
    with st.sidebar:
        st.header("📊 Model Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "95%")
        with col2:
            st.metric("Classes", "8")
        
        st.markdown("**Architecture:** CNN (ResNet50) + Transformer")
        st.markdown("**Framework:** PyTorch")
        
        st.markdown("---")
        st.subheader("🔬 Cancer Types")
        
        with st.expander("🔴 Malignant (Cancerous)", expanded=True):
            for i, cancer_type in enumerate(malignant_types, 1):
                st.markdown(f"{i}. {cancer_type}")
        
        with st.expander("🟢 Benign (Non-Cancerous)", expanded=True):
            for i, cancer_type in enumerate(benign_types, 1):
                st.markdown(f"{i}. {cancer_type}")
        
        st.markdown("---")
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload** a histopathology image
        2. **Click** 'Analyze Image' button
        3. **Review** the classification results
        4. **Check** confidence scores for all types
        """)
        
        st.markdown("---")
        st.caption("Developed with ❤️ using Streamlit & PyTorch")
    
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    uploaded_file = st.file_uploader(
    "📤 Upload Histopathology Image",
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Uploaded Image")
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_column_width=True)
                st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
            except Exception as e:
                st.error(f"❌ Failed to load image: {e}")
                st.stop()

        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            if st.button("🔬 Analyze Image", type="primary"):
                with st.spinner("Analyzing image... Please wait"):
                    try:
                        img_tensor = preprocess_image(image)
                        predicted_class, confidence, all_probs = predict(model, img_tensor, device)
                        predicted_name = class_names[predicted_class]
                        
                        is_malignant = predicted_class < 4
                        
                        if is_malignant:
                            st.error("### 🔴 Malignant (Cancerous) Detected")
                            st.markdown(f"**Detected Type:** {predicted_name}")
                        else:
                            st.success("### 🟢 Benign (Non-Cancerous) Detected")
                            st.markdown(f"**Detected Type:** {predicted_name}")
                        
                        st.metric("Confidence Level", f"{confidence*100:.2f}%")
                        st.progress(confidence)
                        
                        st.session_state['analyzed'] = True
                        st.session_state['all_probs'] = all_probs
                        st.session_state['predicted_name'] = predicted_name
                        st.session_state['is_malignant'] = is_malignant
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.stop()
        
        if st.session_state.get('analyzed', False):
            st.markdown("---")
            st.subheader("📊 Detailed Probability Analysis")
            
            all_probs = st.session_state['all_probs']
            
            fig = create_probability_chart(all_probs, class_names)
            st.plotly_chart(fig, use_column_width=True)
            
            st.subheader("📋 Probability Table")
            
            prob_df = {
                'Cancer Type': class_names,
                'Category': ['🔴 Malignant']*4 + ['🟢 Benign']*4,
                'Probability': [f"{p*100:.2f}%" for p in all_probs],
                'Confidence Bar': all_probs
            }
            
            import pandas as pd
            df = pd.DataFrame(prob_df)
            st.dataframe(
                df[['Cancer Type', 'Category', 'Probability']],
            )
            
            st.info("""
            **🏥 Important Medical Guidelines:**
            
            - This AI analysis is a **supportive tool** and should not replace professional medical examination
            - Always consult with an **oncologist** or **pathologist** for definitive diagnosis
            - Early detection through regular screening saves lives
            - Follow up with your healthcare provider for appropriate treatment plans
            """)
            
            st.warning("""
            **⚠️ Disclaimer:** This model is trained for educational and research purposes. 
            Clinical decisions should never be based solely on AI predictions.
            """)
    
    else:
        st.info("👆 Please upload a histopathology image to begin analysis")
        
        with st.expander("ℹ️ Sample Image Guidelines"):
            st.markdown("""
            **For best results, ensure your image:**
            - Is a clear histopathology slide image
            - Has good lighting and focus
            - Is in JPG, JPEG, or PNG format
            - Shows tissue structure clearly
            """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>CNN + Transformer Hybrid Architecture</strong> | Test Accuracy: 95% | For Research & Educational Use</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
