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
from skimage.measure import find_contours
import imageio
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, SecondaryCaptureImageStorage, ExplicitVRLittleEndian
import tempfile
from datetime import datetime
from tensorflow.keras.models import Model

def create_rtstruct(segmentations, uploaded_file, output_path):
    # Aqui, continua o código para criar o RTSTRUCT
    try:
        # Verificar se uploaded_file é um caminho ou um UploadedFile
        if isinstance(uploaded_file, str) or isinstance(uploaded_file, os.PathLike):
            # Caso seja um caminho (path), carregue diretamente
            original_ds = pydicom.dcmread(uploaded_file, force=True)
        elif isinstance(uploaded_file, UploadedFile):
            # Caso seja um UploadedFile, use o .name para carregá-lo
            if uploaded_file.name.endswith('.dcm'):
                # Se for DICOM, carregue diretamente
                original_ds = pydicom.dcmread(uploaded_file, force=True)
            else:
                # Se não for DICOM, crie um DICOM a partir do arquivo
                original_ds = create_dicom_from_non_dicom(uploaded_file, output_path)
                if original_ds is None:
                    return None  # Se falhar na conversão
        else:
            raise TypeError("Tipo de arquivo desconhecido")

        num_slices = len(segmentations)

        # Criar o cabeçalho DICOM
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

        # Criar o dataset principal
        rtstruct = Dataset()
        rtstruct.file_meta = file_meta

        # Preencher os metadados
        rtstruct.SOPClassUID = pydicom.uid.RTStructureSetStorage
        rtstruct.SOPInstanceUID = pydicom.uid.generate_uid()
        rtstruct.PatientName = original_ds.PatientName if 'PatientName' in original_ds else "PAT_Name"
        rtstruct.PatientID = original_ds.PatientID if 'PatientID' in original_ds else "PAT_ID"
        if (0x0008, 0x0102) not in original_ds:
                original_ds.add_new((0x0008, 0x0102), 'SH', 'DCM')
        rtstruct.StudyInstanceUID = original_ds.StudyInstanceUID if 'StudyInstanceUID' in original_ds else pydicom.uid.generate_uid()
        rtstruct.SeriesInstanceUID = pydicom.uid.generate_uid()
        rtstruct.FrameOfReferenceUID = original_ds.FrameOfReferenceUID if 'FrameOfReferenceUID' in original_ds else pydicom.uid.generate_uid()
        rtstruct.StructureSetLabel = "RTSTRUCT"
        rtstruct.StructureSetDate = datetime.now().strftime("%Y%m%d")
        rtstruct.StructureSetTime = datetime.now().strftime("%H%M%S")

        # Adicionar ROI (região de interesse)
        rtstruct.ROIContourSequence = []
        rtstruct.StructureSetROISequence = []
        rtstruct.RTROIObservationsSequence = []

        # Criar a sequência DeidentificationMethodCodeSequence
        coding_scheme_ident_seq = []
        # Criar o item para a sequência
        item = Dataset()

        coding_scheme_ident_seq = []

        # Primeiro item da Coding Scheme Identification Sequence
        item1 = Dataset()
        item1.CodingSchemeDesignator = 'FMA'
        item1.CodingSchemeUID = '2.16.840.1.113883.6.119'

        # Segundo item da Coding Scheme Identification Sequence
        item2 = Dataset()
        item2.CodingSchemeDesignator = '99VMS_STRUCTCODE'
        item2.CodingSchemeUID = '1.2.246.352.7.3.10'
        item2.CodingSchemeName = 'Structure Codes'
        item2.CodingSchemeResponsibleOrganization = 'Gerado por Modelo'

        # Adicionar os itens à sequência
        coding_scheme_ident_seq.append(item1)
        coding_scheme_ident_seq.append(item2)

        rtstruct.CodingSchemeIdentificationSequence = coding_scheme_ident_seq
    
        # Adicionar os campos no item
        item.CodeValue = '113100'
        item.CodingSchemeDesignator = 'DCM'
        item.CodeMeaning = 'Basic Application Confidentiality Profile'
        # Adicionar o item à sequência
        coding_scheme_ident_seq.append(item)
        # Agora, adicionar a sequência ao DICOM
        rtstruct.DeidentificationMethodCodeSequence = coding_scheme_ident_seq

        roi_name = "Metástase"
        roi_number = 1

        # Adicionar contornos da máscara
        for z_idx, slice_mask in enumerate(segmentations):
            contours = find_contours(slice_mask, level=0.5)
            for contour in contours:
                # Coordenadas do contorno
                contour_data = []
                for y, x in contour:
                    contour_data.extend([x, y, z_idx])  # Coordenadas no formato [x1, y1, z1, x2, ...]

                # Criar uma sequência para o contorno
                contour_item = Dataset()
                contour_item.ContourGeometricType = "CLOSED_PLANAR"
                contour_item.NumberOfContourPoints = len(contour) 
                contour_item.ContourData = contour_data
                rtstruct.ROIContourSequence.append(contour_item)

        # Adicionar estrutura ROI
        roi_item = Dataset()
        roi_item.ROIName = roi_name
        roi_item.ROINumber = roi_number
        roi_item.ReferencedFrameOfReferenceUID = original_ds.FrameOfReferenceUID if 'FrameOfReferenceUID' in original_ds else pydicom.uid.generate_uid()
        rtstruct.StructureSetROISequence.append(roi_item)

        # Adicionar observação ROI
        observation_item = Dataset()
        observation_item.ObservationNumber = roi_number
        observation_item.ReferencedROINumber = roi_number
        observation_item.ROIObservationLabel = roi_name
        rtstruct.RTROIObservationsSequence.append(observation_item)

        # Salvar o RTSTRUCT
        rtstruct.save_as(output_path)
        print(f"RTSTRUCT salvo em {output_path}")

        # Retornar o caminho do arquivo gerado
        return output_path

    except Exception as e:
        print(f"Erro ao gerar o RTSTRUCT: {e}")
        return None  # Se ocorrer um erro, retornar None

def create_dicom_from_non_dicom(uploaded_file, output_path):
    """Função para converter arquivo não DICOM para um DICOM e salvá-lo."""
    # Lidar com diferentes tipos de arquivos
    if uploaded_file.name.endswith(".nii") or uploaded_file.name.endswith(".nii.gz"):
        # Carregar NIfTI
        nifti_img = nib.load(uploaded_file)
        img_data = nifti_img.get_fdata()
        dtype = np.uint8 if img_data.max() <= 255 else np.uint16
        img_data = img_data.astype(dtype)

        # Criar um DICOM básico
        ds = Dataset()
        ds.PixelData = img_data.tobytes()
        ds.Rows, ds.Columns, ds.NumberOfFrames = img_data.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8 if dtype == np.uint8 else 16
        ds.BitsStored = ds.BitsAllocated
        ds.HighBit = ds.BitsAllocated - 1
        ds.PixelRepresentation = 0

    elif uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
        img = Image.open(uploaded_file)
        img_data = np.array(img)

        ds = Dataset()
        ds.PixelData = img_data.tobytes()
        ds.Rows, ds.Columns = img_data.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0

    else:
        st.error("Formato de arquivo não suportado!")
        return None

    # Criar o cabeçalho DICOM
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    ds.file_meta = file_meta

    ds.PatientName = "PAT_Name"
    ds.PatientID = "PAT_ID"
    if (0x0008, 0x0102) not in ds:
        ds.add_new((0x0008, 0x0102), 'SH', 'DCM')
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = "OT"

    # Salvar o DICOM em disco
    ds.save_as(output_path)
    return output_path

def generate_grad_cam(model, image, class_index, layer_name):
    # Cria um modelo que inclui até a camada de interesse
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Passa a imagem pelo modelo para obter as ativações e a predição
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[:, class_index]

    # Calcula os gradientes da predição em relação às ativações da camada
    grads = tape.gradient(loss, conv_outputs)

    # Calcula os pesos (média global dos gradientes para cada filtro)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Combina os pesos com as ativações
    cam = np.dot(conv_outputs[0], weights)

    # Normaliza os valores para ficar entre 0 e 1
    cam = np.maximum(cam, 0)  # Remove valores negativos
    cam = cam / cam.max()  if cam.max() != 0 else 0 # Normaliza para [0, 1]

    return cam

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

#model.load_weights('tcc-app/model_weights.h5')

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


import os
import sys

def resource_path(relative_path):
    """Obter o caminho absoluto para recursos, independente do ambiente"""
    try:
        # Quando o executável é criado, os recursos são movidos para _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # No ambiente de desenvolvimento, usa o diretório tcc-app
        base_path = os.path.abspath("." )#"tcc-app")

    return os.path.join(base_path, relative_path)


# Mapa para pegar os pesos do modelo escolhido pelo usuário
def get_weights_path(tumor_type, modality):
    weights_map = {
        ("Metástase", "T1 com contraste"): resource_path(os.path.join('weights', 'meta_t1c_weights.h5')),
        ("Metástase", "FLAIR"): resource_path(os.path.join('weights', 'meta_flair_weights.h5')),
        ("Metástase", "BRAVO"): resource_path(os.path.join('weights', 'meta_bravo_weights.h5')),
    }
    return weights_map.get((tumor_type, modality), None)


# Sidebar do streamlit

logo_metaseg = Image.open(resource_path(os.path.join('images', 'MetaSeg.png')))

st.sidebar.image(logo_metaseg)

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

    # Armazena o objeto de arquivo completo no session_state
    previous_file_name = st.session_state.get("uploaded_file_name", None)
    
    # Upload do arquivo
    uploaded_file = st.file_uploader("Escolha um arquivo DICOM, NIfTI, JPEG, JPG, PNG ou TIFF", type=["dcm", "nii", "nii.gz", "jpg", "png", "tiff", "jpeg"])
    
    if uploaded_file is not None:
        # Verifica se o arquivo foi alterado
        if uploaded_file.name != previous_file_name:
            # Limpa os dados relacionados à visualização e inferência
            st.session_state.pop("segmentations", None)
            st.session_state.pop("img_data", None)

        # Armazena o arquivo completo e o nome
        st.session_state.uploaded_file = uploaded_file
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")

    else:
        # Limpa os dados se o upload for cancelado
        st.session_state.pop("uploaded_file_name", None)
        st.session_state.pop("uploaded_file", None)
        st.session_state.pop("segmentations", None)
        st.session_state.pop("img_data", None)

    # Seletor de tipo de tumor
    tumor_type = st.selectbox("Selecione o Tipo de Tumor", ["Metástase", "Glioblastoma", "Meningioma"])
    image_modality = st.selectbox("Selecione a Modalidade de Imagem", ["T1 com contraste", "T1 sem contraste", "T2", "FLAIR", "BRAVO"])

# Coluna 2: Inferência
with col2:
    st.header("Inferência e Download")

    # Botão para realizar a inferência
    if st.button("Realizar Inferência"):
        if uploaded_file is not None:
            try:
                weights_path = get_weights_path(tumor_type, image_modality)
                if weights_path and os.path.exists(weights_path):
                    model.load_weights(weights_path)  # Carrega os pesos específicos
                    st.success(f"Realizando inferência, por favor aguarde...")
                else:
                    st.error(f"Pesos não encontrados para {tumor_type} na modalidade {image_modality}.")
                    st.stop()
                    
                progress_bar = st.progress(0)

                # Salvar o arquivo temporariamente
                _, ext = os.path.splitext(uploaded_file.name)
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name

                # Carregar os dados da imagem
                if ext in [".nii", ".nii.gz", ".gz"]:
                    temp_suffix = ".nii.gz" if ext.lower() in [".gz", ".nii.gz"] else ".nii"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name
                    
                    # Carregar imagem NIfTI
                    img = nib.load(tmp_file_path)
                    img_data = img.get_fdata()

                elif ext == ".dcm":
                    # Carregar imagem DICOM
                    dicom_ds = pydicom.dcmread(tmp_file_path, force=True)
                    img_data = dicom_ds.pixel_array

                    if len(img_data.shape) == 3:
                        img_data = np.transpose(img_data, (1, 2, 0))

                    # Adicionar dimensão extra se for 2D (apenas um slice)
                    if len(img_data.shape) == 2:
                        img_data = img_data[:, :, np.newaxis]

                elif ext in [".jpg", ".jpeg", ".png", ".tiff"]:  # Se for uma imagem nesses formatos
                    # Carregar a imagem nos formatos indicados
                    img = Image.open(uploaded_file)
                    img_data = np.array(img)  # Converter para um array NumPy
                    if img_data.ndim == 3 and img_data.shape[2] == 3:  # Se for RGB
                        img_data = np.mean(img_data, axis=2).astype(np.uint8) # Média para levar de RGB para escala de cinza
                    
                    if img_data.ndim != 3:
                        img_data = np.expand_dims(img_data, axis=-1)

                # Realizar a segmentação
                smooth_segmentations = []
                segmentations = []
                num_slices = img_data.shape[2]
                for i in range(num_slices):
                    slice_image = img_data[:, :, i]  # Fatia 2D
                    img_array = preprocess_image(slice_image)
                    model_output = model.predict(img_array)
                    mask_class_1 = model_output[0, :, :, 1]  # Máscara da classe 1
                    mask_resized = resize(
                            mask_class_1, 
                            slice_image.shape,
                            order=0,
                            preserve_range=True, 
                            anti_aliasing=False
                        )

                    smooth_segmentations.append(mask_resized)
                    mask_resized = np.where(mask_resized > 0.5, 1, 0)
                    segmentations.append(mask_resized)
                    progress_bar.progress((i + 1) / num_slices)

                # Armazenar os dados no session_state
                st.session_state.img_data = img_data
                st.session_state.segmentations = segmentations
                st.session_state.smooth_segmentations = smooth_segmentations


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
                    # Verificar se há múltiplos slices
                    num_slices = len(segmentations)

                    # Criar o cabeçalho DICOM
                    file_meta = FileMetaDataset()
                    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
                    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

                    # Criar o dataset principal
                    ds = Dataset()
                    ds.file_meta = file_meta

                    # Adicionar os dados da imagem (multi-slice)
                    stacked_segmentations = np.stack(segmentations, axis=0)  # Empilha todas as segmentações
                    ds.PixelData = (stacked_segmentations * 255).astype(np.uint8).tobytes()  # Converter para uint8 e serializar

                    # Configurar metadados DICOM
                    ds.Rows, ds.Columns = segmentations[0].shape  # Dimensões do slice
                    ds.NumberOfFrames = num_slices  # Número de slices
                    ds.BitsAllocated = 8  # 8 bits por pixel (uint8)
                    ds.BitsStored = 8  # 8 bits efetivamente armazenados
                    ds.HighBit = 7  # Bit mais significativo
                    ds.PixelRepresentation = 0  # 0 para unsigned int
                    ds.SamplesPerPixel = 1  # Imagem monocromática
                    ds.PhotometricInterpretation = "MONOCHROME2"  # Imagem em escala de cinza

                    # Configurar outros metadados DICOM
                    ds.SOPClassUID = SecondaryCaptureImageStorage
                    ds.SOPInstanceUID = pydicom.uid.generate_uid()
                    ds.Modality = "OT"  # Outros
                    ds.SeriesDescription = "Segmentação Gerada por Modelo"
                    ds.ImageType = ["DERIVED", "SECONDARY"]

                    # Outros metadados úteis

                    ds.StudyInstanceUID = pydicom.uid.generate_uid()
                    ds.SeriesInstanceUID = pydicom.uid.generate_uid()

                    if uploaded_file is not None and uploaded_file.name.lower().endswith(".dcm"):
                        # Carregar o arquivo DICOM para extrair o PatientName e PatientID
                        dicom_ds = pydicom.dcmread(uploaded_file, force=True)
                        patient_name = dicom_ds.PatientName if 'PatientName' in dicom_ds else "PAT_Name"
                        patient_id = dicom_ds.PatientID if 'PatientID' in dicom_ds else "PAT_ID"
                    else:
                        # Se não for DICOM, usa valores genéricos
                        patient_name = "PAT_Name"
                        patient_id = "PAT_ID"

                    # Criar buffer para salvar o DICOM
                    dicom_buffer = BytesIO()

                    # Salvar o dataset como arquivo DICOM
                    pydicom.filewriter.write_file(dicom_buffer, ds)

                    # Exibir para download no Streamlit
                    st.download_button(
                        label="Baixar Máscara Multi-slice em DICOM",
                        data=dicom_buffer.getvalue(),
                        file_name="mask_segmentation_multislice.dcm",
                        mime="application/dicom"
                    )

                    # Após o DICOM estar pronto, gerar o RTSTRUCT
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_path = temp_file.name

                    # Gerar o RTSTRUCT
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_rtstruct:
                        tmp_rtstruct_path = tmp_rtstruct.name
                        rtstruct_output = create_rtstruct(segmentations, temp_file_path, tmp_rtstruct_path)
                        
                        if rtstruct_output:
                            with open(rtstruct_output, "rb") as rtstruct_file:
                                st.download_button(
                                    label="Baixar RT Structure Set DICOM",
                                    data=rtstruct_file,
                                    file_name="rtstruct_output.dcm",
                                    mime="application/dicom"
                                )
                        else:
                            st.error("Erro ao gerar o RT Structure Set.")


                except Exception as e:
                    st.error(f"Erro ao gerar arquivo DICOM: {str(e)}")

        else:
            st.warning("Por favor, realize a inferência antes de fazer o download.")


# Coluna 3: Visualização
with col3:
    st.header("Visualização do Slice")

    if uploaded_file is not None:
        # Verifica se os dados da inferência estão no session_state
        if 'smooth_segmentations' in st.session_state and 'img_data' in st.session_state:
            smooth_segmentations = st.session_state.smooth_segmentations
            img_data = st.session_state.img_data
            num_slices = len(smooth_segmentations)

            if num_slices > 1:
                # Slider para escolher o slice
                slice_idx = st.slider("Escolha o Slice", 0, num_slices - 1, 0)
            else:
                # Mensagem informando que há apenas uma fatia
                st.info("A imagem é constituída de apenas um slice.")
                slice_idx = 0  # Usar o único slice disponível

            # Redimensiona a imagem original para 256x256 para combinar com a máscara
            slice_image = img_data[:, :, slice_idx]  # Obtém o slice
            image = Image.fromarray(slice_image.astype(np.uint8))  # Converte para PIL
            image = image.resize((256, 256))  # Redimensiona para 256x256
            resized_image = np.array(image)  # Converte de volta para array NumPy

            # Corrige a entrada do modelo cam
            image_cam = np.repeat(resized_image[..., np.newaxis], 3, axis=-1)  # Replica o canal para 3 dimensões

            # Gera o Grad-CAM
            cam_upsample = generate_grad_cam(model, image_cam, 1, 'up_block_4_upsample')
            cam_residual = generate_grad_cam(model, image_cam, 1, 'up_block_4_residual_conv')

            if cam_upsample is None or np.array(cam_upsample).ndim == 0 or cam_upsample.size == 0:
                cam_upsample = np.zeros((256, 256))  # Matriz de zeros pra substituir o None de quando o Grad-CAM não retornar nada

            if cam_residual is None or np.array(cam_residual).ndim == 0 or cam_residual.size == 0:
                cam_residual = np.zeros((256, 256))  # Matriz de zeros pra substituir o None de quando o Grad-CAM não retornar nada

            cam_upsample_resized = resize(
                cam_upsample,
                slice_image.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=True
            )

            cam_residual_resized = resize(
                cam_residual,
                slice_image.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=True
            )

            image_with_mask = slice_image * smooth_segmentations[slice_idx]
            mask_complement = 1 - smooth_segmentations[slice_idx]
            image_without_mask = slice_image * mask_complement

            # Visualiza a imagem original redimensionada
            st.subheader("Imagem Original")
            fig, ax = plt.subplots()
            ax.imshow(slice_image, cmap='gray')
            ax.set_title(f"Slice {slice_idx} - Imagem Original")
            st.pyplot(fig)

            # Visualiza a máscara gerada pelo modelo
            st.subheader("Máscara de Segmentação")
            fig, ax = plt.subplots()
            #ax.imshow(cam_upsample_resized, cmap='gnuplot', alpha=0.75)
            #ax.imshow(cam_residual_resized, cmap='gnuplot', alpha=0.75)
            ax.imshow(smooth_segmentations[slice_idx], cmap='inferno', alpha=0.5)
            ax.imshow(image_with_mask, cmap='copper', alpha=0.9)
            ax.imshow(image_without_mask, cmap='gray', alpha=0.15)
            ax.set_title(f"Slice {slice_idx} - Máscara de Segmentação \n Imagem original em transparência")
            st.pyplot(fig)
        else:
            st.warning("Por favor, realize a inferência antes de visualizar os slices.")
    else:
        st.warning("Faça o upload de uma imagem para começar.")
