import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from io import BytesIO
import tempfile
import tarfile
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers, initializers
import random
import segmentation_models as sm
import pandas as pd
import re
from PIL import Image
from scipy.ndimage import distance_transform_edt as distance
from skimage.transform import resize
import imageio
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset


img_size = (256, 256)
num_classes = 2

os.environ["SM_FRAMEWORK"] = 'tf.keras'

def reset_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

reset_seeds()

initializer = initializers.RandomNormal(stddev=0.01)

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,), name="input_layer")

    # Entrada
    x = layers.Conv2D(32, 3, strides=2, padding="same", name="entry_conv")(inputs)
    x = layers.BatchNormalization(name="entry_bn")(x)
    x = layers.Activation("relu", name="entry_activation")(x)

    previous_block_activation = x  # Set aside residual

    # Blocos de downsampling
    for i, filters in enumerate([64, 128, 256], start=1):
        x = layers.Activation("relu", name=f"down_block_{i}_activation1")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", name=f"down_block_{i}_sepconv1")(x)
        x = layers.BatchNormalization(name=f"down_block_{i}_bn1")(x)

        x = layers.Activation("relu", name=f"down_block_{i}_activation2")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", name=f"down_block_{i}_sepconv2")(x)
        x = layers.BatchNormalization(name=f"down_block_{i}_bn2")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same", name=f"down_block_{i}_maxpool")(x)

        # Residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same", name=f"down_block_{i}_residual")(previous_block_activation)
        x = layers.add([x, residual], name=f"down_block_{i}_add")
        previous_block_activation = x

    # Blocos de upsampling
    for i, filters in enumerate([256, 128, 64, 32], start=1):
        x = layers.Activation("relu", name=f"up_block_{i}_activation1")(x)
        x = layers.Dropout(0.1, name=f"up_block_{i}_dropout1")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same", name=f"up_block_{i}_convT1")(x)
        x = layers.BatchNormalization(name=f"up_block_{i}_bn1")(x)

        x = layers.Activation("relu", name=f"up_block_{i}_activation2")(x)
        x = layers.Dropout(0.1, name=f"up_block_{i}_dropout2")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same", name=f"up_block_{i}_convT2")(x)
        x = layers.BatchNormalization(name=f"up_block_{i}_bn2")(x)

        x = layers.UpSampling2D(2, name=f"up_block_{i}_upsample")(x)

        # Residual
        residual = layers.UpSampling2D(2, name=f"up_block_{i}_residual_upsample")(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same", name=f"up_block_{i}_residual_conv")(residual)
        x = layers.add([x, residual], name=f"up_block_{i}_add")
        previous_block_activation = x

    # Camada final de classificação por pixel
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same", name="output_layer")(x)

    # Define o modelo
    model = keras.Model(inputs, outputs, name="segmentation_model")
    return model


class SurfaceLoss(tf.keras.losses.Loss):
    def __init__(self, name="surface_loss"):
        super().__init__(name=name)

    def calc_dist_map(self, seg):
        res = np.zeros_like(seg)
        posmask = seg.astype(bool)

        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

        return res

    def calc_dist_map_batch(self, y_true):
        y_true_numpy = y_true.numpy()
        return np.array([self.calc_dist_map(y) for y in y_true_numpy]).reshape(y_true.shape).astype(np.float32)

    def call(self, y_true, y_pred):
        y_true_dist_map = tf.py_function(func=self.calc_dist_map_batch,
                                         inp=[y_true],
                                         Tout=tf.float32)
        multipled = y_pred * y_true_dist_map
        return tf.reduce_mean(multipled, axis=-1)

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, loss_1, loss_2, loss_3, name="combined_loss"):
        super().__init__(name=name)
        self.loss_1 = loss_1
        self.loss_2 = loss_2
        self.loss_3 = loss_3

    def call(self, y_true, y_pred):
        first_loss = self.loss_1(y_true, y_pred)
        second_loss = self.loss_2(y_true, y_pred)
        third_loss = self.loss_3(y_true, y_pred)
        return first_loss + second_loss + third_loss


weights = [0.2]
eta = [9]
psi = [2]

def binary_crossentropy_loss(y_true, y_pred):
    loss_fn = sm.losses.BinaryCELoss()
    return loss_fn(y_true, y_pred)

def weighted_tversky_loss(y_true, y_pred):
    loss_fn = sm.losses.WeightedTverskyLoss(class_weights=weights, eta=eta, psi=psi)
    return loss_fn(y_true, y_pred)

def surface_loss(y_true, y_pred):
    loss_fn = SurfaceLoss()  # Supondo que você tenha uma implementação de SurfaceLoss
    return loss_fn(y_true, y_pred)

def get_combined_loss():
    return CombinedLoss(binary_crossentropy_loss, weighted_tversky_loss, surface_loss)

model = get_model(img_size, num_classes)

learning_rate = 1e-3
total_loss = CombinedLoss(binary_crossentropy_loss, weighted_tversky_loss, surface_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), sm.metrics.Precision(threshold=0.5), sm.metrics.Recall(threshold=0.5)]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=total_loss,
    metrics=metrics
)

model.load_weights('tcc-app/model_weights.h5')

# Função de exemplo para o plot 3D (a lógica de plotagem será adaptada)
def plot_3d_image(image_data):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(range(image_data.shape[0]), range(image_data.shape[1]))
    ax.plot_surface(x, y, image_data[:, :, int(image_data.shape[2] / 2)], cmap='viridis')
    plt.title("Plot 3D do Volume")
    st.pyplot(fig)


def preprocess_image(image):
    # Certifique-se de que `image` seja um array NumPy 2D
    if len(image.shape) == 2:  # Verifica se é uma imagem 2D
        image = np.expand_dims(image, axis=-1)  # Adiciona uma dimensão de canal (fica com shape (256, 256, 1))
        image = np.tile(image, (1, 1, 3))  # Duplica o canal para criar 3 canais (fica com shape (256, 256, 3))
    image = Image.fromarray(image.astype(np.uint8))  # Certifique-se de que seja uma imagem PIL
    image = image.resize((256, 256))  # Redimensiona para 256x256
    image_array = np.array(image)  # Converte para array NumPy
    image_array = np.expand_dims(image_array, axis=0)  # Adiciona a dimensão do batch (fica com shape (1, 256, 256, 3))
    image_array = image_array / 255.0  # Normaliza para [0, 1]
    return image_array


# Sidebar do streamlit
st.sidebar.image("tcc-app/MetaSeg.png")
st.sidebar.markdown(
    """
    Acesse o [GitHub](https://github.com/brunotakara/tcc-app) do projeto
    """
)

# Texto sobre o desenvolvedor
st.sidebar.write("Desenvolvido por [Bruno Takara](https://www.linkedin.com/in/bruno-yukio-takara-268486114/) para o Trabalho de Conclusão de Curso de Física Médica pela UFCSPA")

# Layout com 3 colunas
col1, col2, col3 = st.columns(3)

# Coluna 1: Upload e Configurações
with col1:
    st.header("Arquivos")
    
    # Upload do arquivo
    uploaded_file = st.file_uploader("Escolha um arquivo DICOM ou NIfTI", type=["dcm", "nii", "nii.gz"])
    
    if uploaded_file is not None:
        # Salvar o nome do arquivo carregado no session_state
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")

    # Seletor de tipo de tumor
    tumor_type = st.selectbox("Selecione o Tipo de Tumor", ["Metástase", "Glioblastoma", "Meningioma"])
    image_modality = st.selectbox("Selecione a Modalidade de Imagem", ["T1", "T1 com contraste", "T2", "T2 FLAIR", "BRAVO"])

# Coluna 2: Inferência
with col2:
    st.header("Inferência e Download")

    # Botão para realizar a inferência
    if st.button("Realizar Inferência"):
        if uploaded_file is not None:
            try:
                progress_text = "Realizando inferência, por favor aguarde..."
                progress_bar = st.progress(0)

                # Salvar o arquivo temporariamente
                _, ext = os.path.splitext(uploaded_file.name)
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name

                # Carregar os dados da imagem
                img = nib.load(tmp_file_path)
                img_data = img.get_fdata()

                # Realizar a segmentação
                segmentations = []
                num_slices = img_data.shape[2]
                for i in range(num_slices):
                    slice_image = img_data[:, :, i]  # Fatia 2D
                    img_array = preprocess_image(slice_image)
                    model_output = model.predict(img_array)
                    mask_class_1 = model_output[0, :, :, 1]  # Máscara da classe 1
                    segmentations.append(mask_class_1)
                    progress_bar.progress((i + 1) / num_slices)

                # Armazenar os dados no session_state
                st.session_state.img_data = img_data
                st.session_state.segmentations = segmentations

                st.success("Inferência concluída com sucesso!")
            except Exception as e:
                st.error(f"Erro durante a inferência: {str(e)}")
        else:
            st.error("Faça o upload de uma imagem antes de realizar a inferência.")

    # Seleção do formato de saída
    output_format = st.selectbox("Selecione o formato do output", ["NIfTI", "PNG", "DICOM"])

    if st.button("Preparar para Download"):
        if 'segmentations' in st.session_state:
            segmentations = st.session_state.segmentations
            num_slices = len(segmentations)

            # Verifica o formato selecionado para download
            if output_format == "PNG":
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Salvar cada slice como PNG
                    for i, mask in enumerate(segmentations):
                        mask_image = (mask * 255).astype(np.uint8)  # Converte para 0-255
                        mask_filename = os.path.join(tmpdirname, f"slice_{i+1}.png")
                        imageio.imwrite(mask_filename, mask_image)

                    # Compactar os PNGs em um único arquivo .tar.gz
                    tar_gz_buffer = BytesIO()
                    with tarfile.open(fileobj=tar_gz_buffer, mode="w:gz") as tar:
                        for png_file in os.listdir(tmpdirname):
                            tar.add(os.path.join(tmpdirname, png_file), arcname=png_file)

                    # Preparar o download do arquivo compactado
                    st.download_button(
                        label="Baixar Máscaras em PNG (zip)",
                        data=tar_gz_buffer.getvalue(),
                        file_name="masks.tar.gz",
                        mime="application/gzip"
                    )

            elif output_format == "NIfTI":
                # Cria um arquivo NIfTI com as máscaras de segmentação
                segmentation_array = np.stack(segmentations, axis=-1)  # Empilha as máscaras
                mask_img = nib.Nifti1Image(segmentation_array, np.eye(4))  # Cria uma imagem NIfTI

                # Salva o arquivo NIfTI temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_nifti:
                    tmp_nifti_path = tmp_nifti.name
                    nib.save(mask_img, tmp_nifti_path)

                # Fornece o arquivo NIfTI para o download
                with open(tmp_nifti_path, "rb") as nifti_file:
                    st.download_button(
                        label="Baixar Máscaras em NIfTI",
                        data=nifti_file,
                        file_name="mask_output.nii.gz",
                        mime="application/gzip"
                    )

            elif output_format == "DICOM":
                try:
                    # Criar um arquivo DICOM com a máscara segmentada
                    ds = Dataset()
                    ds.PixelData = (segmentations[0] * 255).astype(np.uint8).tobytes()  # Usa o primeiro slice como exemplo

                    # Definir os metadados do DICOM
                    ds.Rows, ds.Columns = segmentations[0].shape
                    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
                    ds.SOPInstanceUID = pydicom.uid.generate_uid()
                    ds.Modality = "OT"  # Outros
                    ds.SeriesDescription = "Segmentação Gerada por Modelo"
                    ds.ImageType = ["DERIVED", "SECONDARY"]
                    ds.is_little_endian = True
                    ds.is_implicit_VR = True

                    # Salvar o arquivo DICOM
                    dicom_buffer = BytesIO()
                    pydicom.filewriter.write_file(dicom_buffer, ds)

                    # Disponibilizar para download
                    st.download_button(
                        label="Baixar Máscara em DICOM",
                        data=dicom_buffer.getvalue(),
                        file_name="mask_segmentation.dcm",
                        mime="application/dicom"
                    )

                except Exception as e:
                    st.error(f"Erro ao gerar arquivo DICOM: {str(e)}")

        else:
            st.warning("Por favor, realize a inferência antes de fazer o download.")


# Coluna 3: Visualização
with col3:
    st.header("Visualização do Slice")

    if uploaded_file is not None:
        # Verifica se os dados da inferência estão no session_state
        if 'segmentations' in st.session_state and 'img_data' in st.session_state:
            segmentations = st.session_state.segmentations
            img_data = st.session_state.img_data
            num_slices = len(segmentations)

            # Slider para escolher o slice
            slice_idx = st.slider("Escolha o Slice", 0, num_slices - 1, 0)

            # Redimensiona a imagem original para 256x256 para combinar com a máscara
            slice_image = img_data[:, :, slice_idx]  # Obtém o slice
            image = Image.fromarray(slice_image.astype(np.uint8))  # Converte para PIL
            image = image.resize((256, 256))  # Redimensiona para 256x256
            resized_image = np.array(image)  # Converte de volta para array NumPy

            # Visualiza a imagem original redimensionada
            st.subheader("Imagem Original")
            fig, ax = plt.subplots()
            ax.imshow(slice_image, cmap='gray')
            ax.set_title(f"Slice {slice_idx} - Imagem Original (Redimensionada)")
            st.pyplot(fig)

            # Visualiza a máscara gerada pelo modelo
            st.subheader("Máscara de Segmentação")
            fig, ax = plt.subplots()
            ax.imshow(segmentations[slice_idx], cmap='jet', alpha=0.75)
            ax.imshow(resized_image, cmap='gray', alpha=0.25)
            ax.set_title(f"Slice {slice_idx} - Máscara de Segmentação \n (redimensionada e sobreposta)")
            st.pyplot(fig)
        else:
            st.warning("Por favor, realize a inferência antes de visualizar os slices.")
    else:
        st.warning("Faça o upload de uma imagem para começar.")
