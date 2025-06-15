import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
except ImportError:
    faiss = None


# Классы для SimCLR (нужны для загрузки модели)
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.projector = ProjectionHead(2048)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return z


# Кэшируем загрузку моделей
@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt')


@st.cache_resource
def load_simclr_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimCLR()
    checkpoint = torch.load('simclr_cow_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)


@st.cache_data
def load_embeddings():
    return np.load("cow_embeddings.npy", allow_pickle=True).item()


# Функция для получения эмбеддинга изображения
def get_embedding(model, image):
    device = next(model.parameters()).device
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.backbone(img_tensor).squeeze().cpu().numpy()
        return emb / np.linalg.norm(emb)


# Функция поиска похожих коров
def find_similar_cows(query_embedding, embeddings_db, top_k=3):
    query_vector = query_embedding.reshape(1, -1)
    names = list(embeddings_db.keys())
    all_vectors = np.stack([embeddings_db[name] for name in names])

    if faiss is not None:
        index = faiss.IndexFlatIP(all_vectors.shape[1])
        index.add(all_vectors.astype('float32'))
        D, I = index.search(query_vector.astype('float32'), top_k)

        results = []
        for i, idx in enumerate(I[0]):
            results.append({
                'name': names[idx],
                'similarity': float(D[0][i])
            })
        return results
    else:
        sims = cosine_similarity(query_vector, all_vectors)[0]
        top_idxs = sims.argsort()[::-1][:top_k]

        results = []
        for idx in top_idxs:
            results.append({
                'name': names[idx],
                'similarity': float(sims[idx])
            })
        return results


def convert_coordinates(xywh, img_width, img_height):
    x_center, y_center, w, h = xywh
    left = (x_center - w / 2) * img_width
    right = (x_center + w / 2) * img_width
    top = (y_center - h / 2) * img_height
    bottom = (y_center + h / 2) * img_height
    return int(left), int(top), int(right), int(bottom)


# Загружаем модели
yolo_model = load_yolo_model()
simclr_model = load_simclr_model()
embeddings_db = load_embeddings()

st.title("🐮 Детектор и поиск похожих коров")
st.markdown("Загрузите фото коровы для обнаружения носа и поиска похожих коров в базе")

uploaded_file = st.file_uploader(
    "Выберите изображение...",
    type=["jpg", "jpeg", "png"],
    help="Максимальный размер файла 200MB"
)

if uploaded_file is not None:
    st.subheader("Загруженное изображение")
    original_image = Image.open(uploaded_file).convert("RGB")
    original_image = ImageOps.exif_transpose(original_image)
    st.image(original_image, use_column_width=True)

    if st.button("🔍 Найти нос и похожих коров", type="primary"):
        with st.spinner("Обработка изображения..."):
            # Конвертация в numpy array
            img_array = np.array(original_image)

            # Детекция носа
            results = yolo_model.predict(img_array)

            if len(results[0].boxes) == 0:
                st.error("Нос не обнаружен! Попробуйте другое изображение")
                st.stop()

            # Получаем первый обнаруженный нос
            xywhn = results[0].boxes.xywhn[0].cpu().numpy()
            img_width, img_height = original_image.size

            # Конвертация координат
            left, top, right, bottom = convert_coordinates(
                xywhn,
                img_width,
                img_height
            )

            # Обрезка изображения носа
            cropped_nose = original_image.crop((left, top, right, bottom))

            # Получение эмбеддинга носа
            nose_embedding = get_embedding(simclr_model, cropped_nose)

            # Поиск похожих коров
            similar_cows = find_similar_cows(nose_embedding, embeddings_db, top_k=3)

            # Отображение результатов
            st.subheader("Результаты обработки")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Обнаруженный нос**")
                st.image(cropped_nose, use_column_width=True)

            with col2:
                st.markdown("**Координаты носа**")
                st.json({
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": right - left,
                    "height": bottom - top
                })

            # Результаты поиска похожих коров
            st.subheader("🎯 Топ-3 похожих коровы в базе")

            cols = st.columns(3)
            for i, cow in enumerate(similar_cows):
                with cols[i]:
                    st.markdown(f"**#{i + 1}: {cow['name']}**")
                    st.markdown(f"**Схожесть: {cow['similarity']:.4f}**")

                    st.image(f"cow_f_crop/{cow['name']}")

                    if cow['similarity'] > 0.9:
                        st.success("Очень высокое сходство!")
                    elif cow['similarity'] > 0.8:
                        st.info("Хорошее сходство")
                    else:
                        st.warning("Умеренное сходство")

        st.success("Обработка завершена успешно!")

        with st.expander("ℹ️ Подробная информация"):
            st.write(f"**Количество коров в базе:** {len(embeddings_db)}")
            st.write(f"**Размер эмбеддинга:** {nose_embedding.shape[0]} измерений")
            st.write("**Алгоритм поиска:** " + ("FAISS" if faiss else "Cosine Similarity"))
