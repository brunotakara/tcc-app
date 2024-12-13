from datetime import datetime
import imageio
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import pandas as pd
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, SecondaryCaptureImageStorage, ExplicitVRLittleEndian
import random
import re
from scipy.ndimage import distance_transform_edt as distance
import segmentation_models as sm
from skimage.measure import find_contours
from skimage.transform import resize
import streamlit as st
import tarfile
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
import time
import sys

def create_rtstruct(segmentations, uploaded_file, output_path):
    # O código para criar o RTSTRUCT
    try:
        # Verificar se uploaded_file é um caminho ou um UploadedFile
        if isinstance(uploaded_file, str) or isinstance(uploaded_file, os.PathLike):
            # Caso seja um caminho (path), carregue diretamente
            original_ds = pydicom.dcmread(uploaded_file, force=True)
        elif isinstance(uploaded_file, UploadedFile):
            # Caso seja um UploadedFile, use o .name para carregá-lo
            if uploaded_file.name.endswith('.dcm'):
                # Se for DICOM, carrega diretamente
                original_ds = pydicom.dcmread(uploaded_file, force=True)
            else:
                # Se não for DICOM, cria um DICOM a partir do arquivo
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
        print(f"RTSTRUCT temporário salvo em {output_path}")

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

# Mapa para pegar os pesos do modelo escolhido pelo usuário
def get_weights_path(tumor_type, modality, selected_model):

    if selected_model == 'GTV-MetNet (Takara et al., 2024)':
        weights_map = {
            ("Metástase", "T1 com contraste"): resource_path(os.path.join('weights', 'meta_t1c_weights.h5')),
            ("Metástase", "FLAIR"): resource_path(os.path.join('weights', 'meta_flair_weights.h5')),
            ("Metástase", "BRAVO"): resource_path(os.path.join('weights', 'meta_bravo_weights.h5')),
        }
        return weights_map.get((tumor_type, modality), None)

    if selected_model == 'Met-Seg (Ottesen et al., 2020)':
        weights_map = {
            ("Metástase", "T1 com contraste"): resource_path(os.path.join('weights', '3d_model.pth')),
            ("Metástase", "FLAIR"): resource_path(os.path.join('weights', '3d_model.pth')),
            ("Metástase", "BRAVO"): resource_path(os.path.join('weights', '3d_model.pth')),
            ("Metástase", "T1 sem contraste"): resource_path(os.path.join('weights', '3d_model.pth')),
        }
        return weights_map.get((tumor_type, modality), None)

    if selected_model == 'AURORA (Buchner et al., 2023)':
        weights_map = {
            ("Metástase", "T1 com contraste"): resource_path(os.path.join('weights', 't1c-o_best.tar')),
            ("Metástase", "T1 sem contraste"): resource_path(os.path.join('weights', 't1-o_best.tar')),
            ("Metástase", "FLAIR"): resource_path(os.path.join('weights', 'fla-o_best.tar')),
        }
        return weights_map.get((tumor_type, modality), None)

################################# Modelo GTV - MetNet ###################################

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

def load_model_or_weights(model_path, weights_path, img_size, num_classes):
    try:
        # Tenta carregar o modelo completo
        model = tf.keras.models.load_model(model_path)
        st.success("Modelo carregado com sucesso!")
    except Exception as e:
        st.warning(f"O arquivo do modelo está incompleto. Tentando carregar os pesos...")
        try:
            # Caso falhe, tenta carregar apenas os pesos
            model = get_model(img_size, num_classes)  # Defina a arquitetura aqui
            model.load_weights(weights_path)
            st.success("Pesos carregados com sucesso! Realizando a predição, por favor aguarde...")
        except Exception as e:
            st.error(f"Erro ao carregar os pesos: {str(e)}. Por favor, insira um arquivo Python com a definição da arquitetura do modelo.")

            # Adiciona o botão para upload do arquivo Python com a arquitetura do modelo
            uploaded_architecture = st.file_uploader("Carregar Arquivo Python com Arquitetura do Modelo", type=["py"])

            if uploaded_architecture is not None:
                try:
                    # Salva o arquivo temporariamente para importar
                    architecture_path = os.path.join("temp", uploaded_architecture.name)
                    with open(architecture_path, "wb") as f:
                        f.write(uploaded_architecture.getbuffer())

                    # Importa a arquitetura do modelo do arquivo Python
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("model_architecture", architecture_path)
                    model_architecture = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(model_architecture)

                    # Agora espera-se que o arquivo tenha uma função chamada 'get_model' para definir a arquitetura
                    model = model_architecture.get_model(img_size, num_classes)
                    model.load_weights(weights_path)
                    st.success("Arquitetura carregada com sucesso e pesos aplicados!")
                except Exception as e:
                    st.error(f"Erro ao carregar a arquitetura: {str(e)}. Verifique o arquivo e tente novamente.")
                    st.stop()
    
    return model

def preprocess_image(image):
    # Certifica de que `image` seja um array NumPy 2D
    if len(image.shape) == 2:  # Verifica se é uma imagem 2D
        image = np.expand_dims(image, axis=-1)  # Adiciona uma dimensão de canal (fica com shape (256, 256, 1))
        image = np.tile(image, (1, 1, 3))  # Duplica o canal para criar 3 canais (fica com shape (256, 256, 3))
    image = Image.fromarray(image.astype(np.uint8))  # Certifica de que seja uma imagem PIL
    image = image.resize((256, 256))  # Redimensiona para 256x256
    image_array = np.array(image)  # Converte para array NumPy
    image_array = np.expand_dims(image_array, axis=0)  # Adiciona a dimensão do batch (fica com shape (1, 256, 256, 3))
    image_array = image_array / 255.0  # Normaliza para [0, 1]
    return image_array

def resource_path(relative_path):
    """Obter o caminho absoluto para recursos, independente do ambiente"""
    try:
        # Quando o executável é criado, os recursos são movidos para _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # No ambiente de desenvolvimento, usa o diretório tcc-app
        base_path = os.path.abspath("." )#"tcc-app")

    return os.path.join(base_path, relative_path)



################################# Modelo MetSeg ###################################
from typing import List, Union, Optional, Tuple

import os
from pathlib import Path

from skimage import measure
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

import monai
import nibabel as nib
import numpy as np

from dataclasses import dataclass

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

@dataclass
class StageArgs:
    num_modules: int
    num_branches: int
    num_blocks: List[int]
    num_channels: List[int]
    block: str

hrnet_w48 = [
    StageArgs(num_modules=1, num_branches=1, num_blocks=[1], num_channels=[64], block="BASIC"),
    StageArgs(num_modules=1, num_branches=2, num_blocks=[1, 1], num_channels=[48, 96], block="BASIC"),
    StageArgs(num_modules=1, num_branches=3, num_blocks=[1, 1, 1], num_channels=[48, 96, 192], block="BASIC"),
    StageArgs(num_modules=1, num_branches=4, num_blocks=[2, 2, 2, 2], num_channels=[48, 96, 192, 384], block="BASIC"),
    ]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio: float = 1./16,
        pool_types: List[str] = ['avg', 'max'],
        **kwargs,
        ):
        super().__init__()

        self.channels = channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channels, int(channels*ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channels*ratio), channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False),
            nn.InstanceNorm3d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM3D(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio: float = 1./16,
        **kwargs,
        ):
        super().__init__()
        pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(channels=channels, ratio=ratio, pool_types=pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x: torch.Tensor):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        bias: bool = True,
        stride: int = 1,
        ratio: float = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        downsample: nn.Module = None,
        ):
        super().__init__()
        self.ratio = ratio

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = nn.InstanceNorm3d(planes, affine=bias)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = nn.InstanceNorm3d(planes, affine=bias)
        if self.ratio is not None:
            self.attention = CBAM3D(
                channels=planes*self.expansion,
                ratio=ratio,
            )


        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.ratio is not None:
            out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        bias: bool = True,
        stride: int = 1,
        ratio: float = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        downsample: nn.Module = None,
        ):
        super().__init__()
        self.ratio = ratio

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(planes, affine=bias)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(planes, affine=bias)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.InstanceNorm3d(planes * self.expansion, affine=bias)
        if self.ratio is not None:
            self.attention = CBAM3D(
                channels=planes*self.expansion,
                ratio=ratio,
            )

        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.ratio is not None:
            out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.activation(out)

        return out


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class UNetDecoder(nn.Module):
    def __init__(
        self,
        num_channels: List[int],
        activation: nn.Module = nn.ReLU(inplace=True),
        bias: bool = True,
        ):
        super().__init__()
        self.bias = bias
        self.activation = activation
        up_convs = list()
        double_convs = list()

        for i in range(1, len(num_channels)):
            up_convs.append(self._make_upscale_conv(in_channels=num_channels[-i], out_channels=num_channels[-i-1]))
            double_convs.append(self._make_double_conv(channels=num_channels[-i-1]))

        self.up_convs = nn.ModuleList(up_convs)
        self.double_convs = nn.ModuleList(double_convs)

    def _make_upscale_conv(self, in_channels: int, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(num_features=out_channels, affine=self.bias),
            self.activation,
        )

    def _make_double_conv(self, channels: int):
        return nn.Sequential(
            nn.Conv3d(in_channels=int(2*channels), out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=channels, affine=self.bias),
            self.activation,
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=channels, affine=self.bias),
            self.activation,
        )

    def forward(self, x: List[torch.Tensor]):

        x0 = F.interpolate(x[-1], size=(x[-2].shape[-3], x[-2].shape[-2], x[-2].shape[-1]),
                        mode='trilinear', align_corners=ALIGN_CORNERS)

        x0 = self.up_convs[0](x0)

        for i in range(1, len(x)):
            x0 = self.double_convs[i-1](torch.cat([x0, x[-i-1]], dim=1))
            if i < len(x) - 1:
                shape = x[-i-2].shape
                x0 = F.interpolate(x[-i - 1], size=(shape[-3], shape[-2], shape[-1]),
                    mode='trilinear', align_corners=ALIGN_CORNERS)
                x0 = self.up_convs[i](x0)
        return x0

class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: nn.Module,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        multi_scale_output=True,
        ratio: Union[float, None] = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        bias: bool = True,
        ):

        super().__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.ratio = ratio
        self.bias = bias

        self.multi_scale_output = multi_scale_output

        self.activation = activation
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(
        self,
        num_branches: int,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        ):
        """
        Checks whether the inputs are correct
        """
        
    def _make_one_branch(
        self,
        branch_index: int,
        block: nn.Module,
        num_blocks: int,
        num_channels: int,
        ):
        downsample = None
        if self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            # print(self.num_inchannels[branch_index], num_channels[branch_index], block)
            # Ensure correct shape for the first set of blocks, this is most commonly used after the bottleneck
            downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.num_inchannels[branch_index],
                    out_channels=num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                nn.InstanceNorm3d(
                    num_features=num_channels[branch_index] * block.expansion,
                    affine=self.bias,
                    ),
            )

        layers = []


        layers.append(block(
            inplanes=self.num_inchannels[branch_index],
            planes=num_channels[branch_index],
            bias=self.bias,
            stride=1,
            ratio=self.ratio,
            activation=self.activation,
            downsample=downsample,
            ))

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion

        for l in range(1, num_blocks[branch_index]):
            layers.append(block(
                inplanes=self.num_inchannels[branch_index],
                planes=num_channels[branch_index],
                bias=self.bias,
                stride=1,
                ratio=self.ratio,
                activation=self.activation,
                downsample=None,  # Check if this is supposed to be None
                ))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _fuse_layer(self, start, end):
        """
        Args:
            start (int): the starting index to sample to
            end (int): the end index the fusing ends at
            The zero'eth index represents the highest resolution
        """
        num_inchannels = self.num_inchannels
        if start == end:
            return nn.Identity()
        elif start > end:  # Upsampling
            return nn.Sequential(
                nn.Conv3d(
                    in_channels=num_inchannels[start],
                    out_channels=num_inchannels[end],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.InstanceNorm3d(
                    num_features=num_inchannels[end],
                    affine=self.bias)
                )
        else:
            down_layers = list()
            # Loop from the starting resolution down to the second to bottom resolution, i.e, end - 1
            for _ in range(end - start - 1):
                down_layers.append(nn.Sequential(
                    nn.Conv3d(
                        in_channels=num_inchannels[start],
                        out_channels=num_inchannels[start],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False),
                    nn.InstanceNorm3d(
                        num_features=num_inchannels[start],
                        affine=self.bias),
                    self.activation,
                    ))
            down_layers.append(nn.Sequential(
                nn.Conv3d(
                    in_channels=num_inchannels[start],
                    out_channels=num_inchannels[end],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                nn.InstanceNorm3d(
                    num_features=num_inchannels[end],
                    affine=self.bias),
                ))
            return nn.Sequential(*down_layers)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        fuse_layers = []
        for end in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for start in range(num_branches):
                fuse_layer.append(self._fuse_layer(start=start, end=end))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x: List[torch.Tensor]):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        branches = list()
        for i in range(self.num_branches):
            branches.append(self.branches[i](x[i]))

        fused = list()
        # Comment: The interpolation method work since start_layer is always
        # the up-top most layer from the previous layers
        for end, end_layers in enumerate(self.fuse_layers):
            for start, start_layer in enumerate(end_layers):
                if start == 0:
                    y = start_layer(branches[start])
                else:
                    out = start_layer(branches[start])
                    if start > end:  #  If we have to upsample
                        out = F.interpolate(out, size=[y.shape[-3], y.shape[-2], y.shape[-1]], mode='trilinear', align_corners=ALIGN_CORNERS)
                    y = y + out
            fused.append(self.activation(y))

        return fused

class HighResolutionNet(nn.Module):

    def __init__(
        self,
        config: List[StageArgs],
        inp_classes: int,
        num_classes: int,
        ratio: Union[float, None] = 1./8,
        activation: nn.Module = nn.ReLU(inplace=True),
        bias: bool = True,
        multi_scale_output: bool = True,
        deep_supervision: bool = True,
        ):
        super().__init__()

        self.ratio = ratio

        self.activation = activation
        self.bias = bias
        self.multi_scale_output = multi_scale_output
        self.deep_supervision = deep_supervision

        num_inchannels = [32]
        start_channels = num_inchannels
        self.stem = nn.Sequential(
            nn.Conv3d(inp_classes, num_inchannels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=num_inchannels[0], affine=bias),
            self.activation,
            )

        # """
        # The HR-Net part

        stages = list()
        transitions = list()
        deep_supervision_layers = list()

        for i, stage in enumerate(config):
            # Make HRModule
            new_stage, num_inchannels = self._make_stage(
                layer_config=stage,
                num_inchannels=num_inchannels,
                multi_scale_output=True if i < len(config) - 1 else multi_scale_output,
                )

            stages.append(new_stage)

            # Deep supervision layers
            if i < len(config) - 1:
                deep_supervision_layers.append(
                    nn.Conv3d(
                        in_channels=num_inchannels[0],
                        out_channels=num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                        ))

            # Transition to more resolution layers
            if i < len(config) - 1:
                if stage.num_branches < config[i+1].num_branches:
                    next_block = blocks_dict[config[i+1].block]
                    next_stage_channels = [
                        channels * next_block.expansion for channels in config[i+1].num_channels]

                    transitions.append(self._make_transition_layer(num_inchannels, next_stage_channels))
                    num_inchannels = next_stage_channels


        self.stages = nn.ModuleList(stages)
        self.transitions = nn.ModuleList(transitions)
        if self.deep_supervision:
            self.deep_supervision_layers = nn.ModuleList(deep_supervision_layers)


        self.decoder = UNetDecoder(next_stage_channels, bias=self.bias, activation=self.activation)

        self.output = nn.Conv3d(
                in_channels=next_stage_channels[0],
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                )

    def _make_transition_layer(
        self,
        num_channels_pre_layer: List[int],
        num_channels_cur_layer: List[int],
        ):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv3d(in_channels=num_channels_pre_layer[i],
                                  out_channels=num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.InstanceNorm3d(
                            num_features=num_channels_cur_layer[i],
                            affine=self.bias),
                        self.activation))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv3d(
                            in_channels=inchannels,
                            out_channels=outchannels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
                        nn.InstanceNorm3d(
                            num_features=outchannels,
                            affine=self.bias),
                        self.activation))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config: StageArgs, num_inchannels: int, multi_scale_output=True):
        num_modules = layer_config.num_modules
        num_branches = layer_config.num_branches
        num_blocks = layer_config.num_blocks
        num_channels = layer_config.num_channels
        block = blocks_dict[layer_config.block]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    block=block,
                    num_blocks=num_blocks,
                    num_inchannels=num_inchannels,
                    num_channels=num_channels,
                    multi_scale_output=multi_scale_output,
                    ratio=self.ratio,
                    activation=self.activation,
                    bias=self.bias,
                    ))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x0_h, x0_w, x0_d = x.size(-3), x.size(-2), x.size(-1)

        x = self.stem(x)
        if self.deep_supervision:
            auxiliary = list()

        x = [x]

        for i in range(len(self.stages) - 1):
            x = self.stages[i](x)

            if self.deep_supervision:
                auxiliary.append(self.deep_supervision_layers[i](x[0]))
            transitioned = list()
            for j, transition in enumerate(self.transitions[i]):
                # If length of transitions are larger than the number of previous resolutions
                if j < len(self.transitions[i]) - 1:
                    transitioned.append(transition(x[j]))
                else:
                    transitioned.append(transition(x[-1]))
            x = transitioned

        x = self.stages[-1](x)

        x = self.decoder(x)

        out = self.output(x)

        if self.deep_supervision:
            return out, auxiliary
        return out
        # Previous non-UNet like decoder

        # Divide by 2: 16761MiB
        # Not divide by 2: 33379MiB
        # UNet like decoder: 18095MiB
        x3 = F.interpolate(x[3], size=(x0_h, x0_w, x0_d),
            mode='trilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)
        x0 = F.interpolate(x[0], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x0, x1, x2, x3], 1)


        feats = self.combine(feats)
        # feats = F.interpolate(feats, size=(x0_h, x0_w, x0_d),
                        # mode='trilinear', align_corners=ALIGN_CORNERS)
        # Memory: 4411 if stopped here
        out = self.output(self.activation(feats + x_stem))

        # Memory: 5467 is stopped here
        if self.deep_supervision:
            return out, auxiliary
        return out

# Dummy para verificar output - alterar o valor do out (memória de 27 GB não é suficiente) - descomentar o valor do out para ser a inferência do modelo em vez de inp
def segment_image(file_path, model, device, threshold=0.5):
    # Carregar a imagem NIfTI
    img = nib.load(file_path)
    img_data = img.get_fdata()
    affine = img.affine

    # Expandir dimensões para compatibilidade
    img_data = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)  # Adiciona dimensão de canal

    pred = np.zeros_like(img_data)

    shape = img_data.shape[1:]  # Dimensões espaciais (X, Y, Z)
    
    # Inicializar o tensor de entrada para 4 canais
    inp = torch.zeros((4,) + shape, dtype=torch.float32)

    # Preencher o canal apropriado com base na modalidade
    if image_modality == "BRAVO":
        inp[0] = img_data.squeeze(0)  # Remove a dimensão do canal ao salvar
    elif image_modality == "T1 com contraste":
        inp[1] = img_data.squeeze(0)
    elif image_modality == "T1 sem contraste":
        inp[2] = img_data.squeeze(0)
    elif image_modality == "FLAIR":
        inp[3] = img_data.squeeze(0)

    # Ajustar a imagem conforme os parâmetros de voxel
    x, y, z = img.header["pixdim"][1:4]
    zooms = (x, y, z)

    x = round(shape[0] * x / x_y_thickness)
    y = round(shape[1] * y / x_y_thickness)
    z = round(shape[2] * z / slice_thickness)

    size = (shape[0], shape[1], shape[2])  # O tamanho final do volume


    # Padronizar o input, interpolando se necessário

    inp = inp.unsqueeze(0).to(device)  # Adicionar dimensão do lote (N)
    inp = torch.nn.functional.interpolate(inp, size=size, mode='trilinear', align_corners=False)
    inp = inp.squeeze(0)  # Remover a dimensão do lote após interpolação
    inp = torch.nn.functional.pad(inp, (10, 10, 10, 10, 10, 10))

    print(f"inp = {np.shape(inp)}")

    pred = torch.zeros(inp.shape[1:])
    print(f"pred = {np.shape(pred)}")
    inp = inp.to(device)
    inp, a, b = crop(inp.squeeze(0))
    inp = norm(inp)

    inp = inp * 4
    
    # Inferência com o modelo
    out = inp #inferer(inp.unsqueeze(0), model)[0]
    mini_pred = torch.nn.functional.logsigmoid(out.squeeze(0).squeeze(0)).exp().to('cpu')
    print(np.shape(mini_pred))
    pred[a[0]: b[0], a[1]: b[1], a[2]: b[2]] = mini_pred[1].squeeze(0)
    
    smooth_pred = pred

    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1

    # Marcação das metástases encontradas
    gt_separate, num = measure.label(pred, background=0, connectivity=2, return_num=True)
    pred = pred[10:-10, 10:-10, 10:-10]

    # Interpolação para o tamanho original da imagem
    pred = torch.nn.functional.interpolate(pred.unsqueeze(0).unsqueeze(0), size=shape, mode='nearest').squeeze(0).squeeze(0).numpy()

   
    # Salvar a imagem segmentada
    name = Path(file_path).stem  # Usa o nome do arquivo sem a extensão
    output_file = f"{name}_segmented.nii.gz"
    img_out = nib.Nifti1Image(pred, affine)
    #nib.save(img_out, output_file)
    
    return num, img_out, smooth_pred  # Retorna o número de metástases encontradas, a img_binarizada e a predição suave com valores entre 0 e 1.

def met_seg_process(input_file_or_dir, model, device):
    # Verifique se o input é um arquivo ou diretório
    if os.path.isdir(input_file_or_dir):
        # Se for diretório, lista todos os arquivos .nii ou .nii.gz
        container = [str(Path(input_file_or_dir) / f) for f in os.listdir(input_file_or_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    elif os.path.isfile(input_file_or_dir) and (input_file_or_dir.endswith('.nii') or input_file_or_dir.endswith('.nii.gz')):
        # Se for um arquivo NIfTI único, cria uma lista com esse arquivo
        container = [input_file_or_dir]
    else:
        print("O caminho fornecido não é um diretório ou um arquivo NIfTI válido.")
        return
    
    # Loop sobre todos os arquivos no container
    for file_path in container:
        num_mets = segment_image(file_path, model, device)[0]
        print(f"{file_path}: Número de metástases encontradas: {num_mets}")



############################ AURORA ########################################
#from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

from dataclasses import dataclass

import numpy as np
import torch

from monai.inferers import SlidingWindowInferer
from monai.networks.nets import BasicUNet
from monai.transforms import RandGaussianNoised
from torch.utils.data import DataLoader

from enum import Enum

from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType
from torch.utils.data import Dataset

class InferenceMode(str, Enum):
    """
    Enum representing different modes of inference based on available image inputs.\n
    In General You should aim to use as many modalities as possible to get the best results.
    """
    T1C_O = "t1c-o"
    """T1C is available."""
    FLA_O = "fla-o"
    """FLAIR is available."""
    T1_O = "t1-o"
    """T1 is available."""

class Device(str, Enum):
    """Enum representing device for model inference."""

    CPU = "cpu"
    """Use CPU"""
    GPU = "cuda"
    """Use GPU (CUDA)"""
    AUTO = "auto"
    """Attempt to use GPU, fallback to CPU."""

IMGS_TO_MODE_DICT = {
    (False, True, False, False): InferenceMode.T1C_O,
    (False, False, False, True): InferenceMode.FLA_O,
    (True, False, False, False): InferenceMode.T1_O,
}

class ModelSelection(str, Enum):
    """Enum representing different strategies for model selection."""
    BEST = "best"
    """Select the best performing model."""
    LAST = "last"
    """Select the last model."""
    VANILLA = "vanilla"
    """Select the vanilla model."""


@dataclass
class BaseConfig:
    """Base configuration for the Aurora model inferer."""

    log_level: int = logging.INFO
    """Logging level. Defaults to logging.INFO."""
    device: Device = Device.AUTO
    """Device for model inference. Defaults to Device.AUTO."""
    cuda_devices: str = "0"
    """CUDA devices to use when using CUDA. Defaults to "0"."""

@dataclass
class AuroraInfererConfig(BaseConfig):
    """Configuration for the Aurora model inferer."""

    tta: bool = False #True originalmente era True mas removi para ver se acelera o fluxo
    """Whether to apply test-time augmentations. Defaults to True."""
    sliding_window_batch_size: int = 1
    """Batch size for sliding window inference. Defaults to 1."""
    workers: int = 0
    """Number of workers for data loading. Defaults to 0."""
    threshold: float = 0.5
    """Threshold for binarizing the model outputs. Defaults to 0.5."""
    sliding_window_overlap: float = 0.5
    """Overlap ratio for sliding window inference. Defaults to 0.5."""
    crop_size: Tuple[int, int, int] = (192, 192, 32)
    """Crop size for sliding window inference. Defaults to (192, 192, 32)."""
    model_selection: ModelSelection = ModelSelection.BEST
    """Model selection strategy. Defaults to ModelSelection.BEST."""

MODALITIES = ["t1", "t1c", "t2", "fla"]


class Output(str, Enum):
    """Enum representing different types of output."""

    SEGMENTATION = "segmentation"
    """Segmentation mask"""
    WHOLE_NETWORK = "whole_network"
    """Whole network output."""
    METASTASIS_NETWORK = "metastasis_network"
    """Metastasis network output."""

class ModelHandler:
    """Class for model loading, inference and post processing"""

    def __init__(
        self, config: AuroraInfererConfig, device: torch.device
    ) -> "ModelHandler":
        """Initialize the ModelHandler and download model weights if necessary.

        Args:
            config (AuroraInfererConfig): config
            device (torch.device): torch device

        Returns:
            ModelHandler: ModelHandler instance
        """
        self.config = config
        self.device = device
        # Will be set during infer() call
        self.model = None
        self.inference_mode = None

        # get location of model weights
        self.model_weights_folder = "weights/"

    def load_aurora_model(
        self, inference_mode: InferenceMode, num_input_modalities: int
    ) -> None:
        """Load the model based on the inference mode. Will reuse previously loaded model if inference mode is the same.

        Args:
            inference_mode (InferenceMode): Inference mode
            num_input_modalities (int): Number of input modalities (range 1-4)
        """
        if not self.model or self.inference_mode != inference_mode:
            #logger.info(f"No loaded compatible model found (Switching from {self.inference_mode} to {inference_mode}). Loading Model and weights...")
            self.inference_mode = inference_mode
            self.model = self._load_model(num_input_modalities=num_input_modalities)
            #logger.info(f"Successfully loaded model.")

    def _load_model(self, num_input_modalities: int) -> torch.nn.Module:
        """Internal method to load the Aurora model based on the inference mode.
        Args:
            num_input_modalities (int): Number of input modalities (range 1-4)
        Returns:
            torch.nn.Module: Aurora model.
        """
        # init model
        model = BasicUNet(
            spatial_dims=3,
            in_channels=num_input_modalities,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )
        # load weights
        weights_path = os.path.join(
            self.model_weights_folder,
            f"{self.inference_mode.value}_{self.config.model_selection.value}.tar",
        )
        if not os.path.exists(weights_path):
            raise NotImplementedError(
                f"No weights found for model {self.inference_mode} and selection {self.config.model_selection}. {os.linesep}Available models: {[mode.value for mode in InferenceMode]}"
            )
        
        model = model.to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
        # The models were trained using DataParallel, hence we need to remove the 'module.' prefix
        # for cpu inference to enable checkpoint loading (since DataParallel is not usable for CPU)
        if self.device == torch.device("cpu"):
            if "module." in list(checkpoint["model_state"].keys())[0]:
                checkpoint["model_state"] = {
                    k.replace("module.", ""): v
                    for k, v in checkpoint["model_state"].items()
                }
        else:
            model = torch.nn.parallel.DataParallel(model)
        model.load_state_dict(checkpoint["model_state"])
        return model

    def _apply_test_time_augmentations(
        self, outputs: torch.Tensor, data: Dict, inferer: SlidingWindowInferer
    ) -> torch.Tensor:
        """Apply test time augmentations to the model outputs.

        Args:
            outputs (torch.Tensor): Model outputs.
            data (Dict): Input data.
            inferer (SlidingWindowInferer): Sliding window inferer.

        Returns:
            torch.Tensor: Augmented model outputs.
        """
        n = 1.0
        for _ in range(4):
            # test time augmentations
            _img = RandGaussianNoised(keys="images", prob=1.0, std=0.001)(data)[
                "images"
            ]
            output = inferer(_img, self.model)
            outputs += output
            n += 1.0
            for dims in [[2], [3]]:
                flip_pred = inferer(torch.flip(_img, dims=dims), self.model)
                output = torch.flip(flip_pred, dims=dims)
                outputs += output
                n += 1.0
        outputs /= n
        return outputs

    def _post_process(self, onehot_model_outputs_CHWD: torch.Tensor) -> np.ndarray:
        """Post-process the model outputs to extract only the metastasis network.

        Args:
            onehot_model_outputs_CHWD (torch.Tensor): One-hot encoded model outputs (Channel Height Width Depth).

        Returns:
            np.ndarray: Post-processed metastasis network data.
        """
        # Create segmentations
        activated_outputs = (
            (onehot_model_outputs_CHWD[0][:, :, :, :].sigmoid()).detach().cpu().numpy()
        )
        binarized_outputs = (activated_outputs >= self.config.threshold).astype(np.uint8)
        enhancing_metastasis = binarized_outputs[1]  # Only enhancing metastasis channel

        return enhancing_metastasis

    def _sliding_window_inference(self, data_loader: DataLoader) -> np.ndarray:
        """Perform sliding window inference using monai.inferers.SlidingWindowInferer and return only metastasis network.

        Args:
            data_loader (DataLoader): Data loader.

        Returns:
            np.ndarray: Post-processed metastasis network data.
        """
        inferer = SlidingWindowInferer(
            roi_size=self.config.crop_size,  # = patch_size
            sw_batch_size=self.config.sliding_window_batch_size,
            sw_device=self.device,
            device=self.device,
            overlap=self.config.sliding_window_overlap,
            mode="gaussian",
            padding_mode="replicate",
        )

        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to(self.device)
            # Currently always only 1 batch
            for data in data_loader:
                inputs = data["images"].to(self.device)
                outputs = inferer(inputs, self.model)
                if self.config.tta:
                    #logger.info("Applying test time augmentations")
                    outputs = self._apply_test_time_augmentations(
                        outputs, data, inferer
                    )
                #logger.info("Post-processing data")
                metastasis_network = self._post_process(
                    onehot_model_outputs_CHWD=outputs,
                )

                #logger.info("Returning metastasis network as Numpy array")
                return metastasis_network

    def infer(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """Perform aurora inference on the given data_loader.

        Args:
            data_loader (DataLoader): data loader

        Returns:
            Dict[str, np.ndarray]: Post-processed data
        """
        return self._sliding_window_inference(data_loader=data_loader)

class InferenceDataset(Dataset):
    def __init__(self, image_path: str, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # Inferência de uma única imagem

    def __getitem__(self, idx):
        img = LoadImage(image_only=True)(self.image_path)
        if self.transform:
            img = self.transform(img)
        return {"images": img}

# Configuração dos transformadores
transform = Compose([
    EnsureChannelFirst(),  # Garante que o canal seja a primeira dimensão
    ScaleIntensity(),      # Normaliza os valores de intensidade
    EnsureType()           # Converte para Tensor se necessário
])


################################ Streamlit ###################################

def upload_and_refresh(uploaded_file):
    """
    Função para o usuário receber mensagem de confirmação de carregamento e que limpa a memória caso o upload seja cancelado

    Args:
        uploaded_file: arquivo carregado no streamlit pelo usuário

    """
    # Verifica se um arquivo foi enviado
    if uploaded_file is not None:
        # Verifica se o arquivo enviado é diferente do anterior
        if uploaded_file.name != st.session_state.get("uploaded_file_name", None):
            # Limpa os dados relacionados à visualização e inferência
            st.session_state.pop("segmentations", None)
            st.session_state.pop("img_data", None)

        # Atualiza os estados com o novo arquivo
        st.session_state.uploaded_file = uploaded_file
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")

    else:
        # Se o upload for cancelado, limpa todos os dados relacionados
        st.session_state.pop("uploaded_file_name", None)
        st.session_state.pop("uploaded_file", None)
        st.session_state.pop("segmentations", None)
        st.session_state.pop("img_data", None)


def calculate_dice_coefficient(ground_truth: np.ndarray, segmentation: np.ndarray) -> float:
        """
        Calcula o coeficiente Dice (DSC) entre a segmentação e a GT.
        - Pode ser utilizado para calcular o Dice por Slice ou o Dice volumétrico
        - O valor é definido como 0 caso não haja uma ROI para ser calculada no slice
        - Caso seja calculado no volume completo, o valor é definido como 0 caso não haja metástase presente

        Args:
            ground_truth (np.ndarray): Máscara original.
            segmentation (np.ndarray): Máscara segmentada.

        Returns:
            float: Coeficiente Dice (entre 0 e 1).

        """
        ground_truth = ground_truth.astype(bool)
        segmentation = segmentation.astype(bool)

        intersection = np.logical_and(ground_truth, segmentation).sum()
        total = ground_truth.sum() + segmentation.sum()

        dice = (2 * intersection) / total if total > 0 else 0.0
        return dice

st.set_page_config(layout="wide")  # Configura o layout para expandido

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
col1, col2, col3 = st.columns([1, 1, 1.5])


# Coluna 1: Upload e Configurações
with col1:
    st.header("Preparação")

    model_choice = st.radio(
        "Escolha o tipo de modelo:",
        ("Rede Pré-treinada", "Modelo Próprio"),
        key="model_choice"
    )

    if model_choice == "Rede Pré-treinada":


        selected_model = st.selectbox(
                "Selecione o Modelo:",
                ["GTV-MetNet (Takara et al., 2024)", "Met-Seg (Ottesen et al., 2020)", "AURORA (Buchner et al., 2023)"],
                key="selected_model"
            )

        if selected_model == "GTV-MetNet (Takara et al., 2024)":

            # Upload do arquivo
            uploaded_file = st.file_uploader("Escolha um arquivo DICOM, NIfTI, JPEG, JPG, PNG ou TIFF", type=["dcm", "nii", "nii.gz", "jpg", "png", "tiff", "jpeg"])

            previous_file_name = st.session_state.get("uploaded_file_name", None)

            upload_and_refresh(uploaded_file)
            
            # Configuração para o modelo GTV - MetNet
            img_size = (256, 256)
            num_classes = 2
            os.environ["SM_FRAMEWORK"] = 'tf.keras'
            model = get_model(img_size, num_classes)

        elif selected_model == "Met-Seg (Ottesen et al., 2020)":

            # Upload do arquivo
            uploaded_file = st.file_uploader("Escolha um arquivo NIfTI", type=["nii", "nii.gz"])

            previous_file_name = st.session_state.get("uploaded_file_name", None)

            upload_and_refresh(uploaded_file)


            # Configuração para o modelo Met-Seg
            model = HighResolutionNet(
                config=hrnet_w48,
                inp_classes=4,
                num_classes=1,
                ratio=None,
                activation=nn.SiLU(inplace=True),
                bias=True,
                multi_scale_output=True,
                deep_supervision=True,
            )
            model.deep_supervision = False

            inferer = monai.inferers.SlidingWindowInferer(
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                overlap=0.5,
                mode="gaussian",
                sigma_scale=0.125,
            )

            # Definir o device como GPU ou CPU caso não haja GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Carregar o modelo        
            x_y_thickness = 0.9375
            slice_thickness = 1
            norm = monai.transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
            crop = monai.transforms.CropForeground(margin=2, k_divisible=16, return_coords=True, mode='constant')
            threshold = 0.99
            mets = dict()

        elif selected_model == "AURORA (Buchner et al., 2023)":
            # Upload do arquivo
            uploaded_file = st.file_uploader("Escolha um arquivo NIfTI", type=["nii", "nii.gz"])

            # Armazena o objeto de arquivo completo no session_state
            previous_file_name = st.session_state.get("uploaded_file_name", None)

            upload_and_refresh(uploaded_file)
        
        # Seletor de tipo de tumor
        tumor_type = st.selectbox(
            "Selecione o Tipo de Tumor",
            ["Metástase", "Glioblastoma", "Meningioma"],
            key="pretrained_tumor_type"
        )

        image_modality = st.selectbox(
            "Selecione a Modalidade de Imagem",
            ["T1 com contraste", "T1 sem contraste", "T2", "FLAIR", "BRAVO"],
            key="pretrained_image_modality"
        )

        # Atualiza o session_state
        st.session_state.model_config = {
            "type": "pretrained",
            "tumor_type": tumor_type,
            "modality": image_modality,
            "model": selected_model
        }

    elif model_choice == "Modelo Próprio":

        # Upload do arquivo para inferência
        uploaded_file = st.file_uploader("Escolha um arquivo DICOM, NIfTI, JPEG, JPG, PNG ou TIFF", type=["dcm", "nii", "nii.gz", "jpg", "png", "tiff", "jpeg"])

        previous_file_name = st.session_state.get("uploaded_file_name", None)

        upload_and_refresh(uploaded_file)

        # Upload do modelo de pesos (.h5)
        uploaded_model = st.file_uploader(
            "Faça upload do seu arquivo de pesos (.h5):",
            type=["h5"],
            key="custom_model_upload"
        )

        if uploaded_model:
            st.success("Pesos carregados com sucesso!")

        # Upload do arquivo de arquitetura (.py)
        uploaded_architecture = st.file_uploader(
            "Faça upload da sua arquitetura (.py) com a função get_model e o import das bibliotecas necessárias:",
            type=["py"],
            key="custom_architecture_upload"
        )

        if uploaded_architecture:
            try:
                # Salvar o conteúdo do arquivo carregado
                filename = "uploaded_architecture.py"
                content = uploaded_architecture.read().decode("utf-8")
                print(content)
                with open(filename, "w") as f:
                    f.write(content)

                # Criar um namespace para executar o código
                custom_namespace = {}
                with open(filename, "r") as f:
                    exec(f.read(), custom_namespace)

                print(custom_namespace["get_model"] )

                if "get_model" in custom_namespace:
                    st.success("Arquitetura carregada com sucesso!")
                    get_model_aurora = custom_namespace["get_model"]  # Referência para a função

            except Exception as e:
                st.error(f"Erro ao processar o arquivo: {e}")       

        # Flag para controlar a exibição da janela de exemplo
        if 'popup_open' not in st.session_state:
            st.session_state.popup_open = False

        # Flag para verificar se a arquitetura foi carregada
        if 'architecture_loaded' not in st.session_state:
            st.session_state.architecture_loaded = False

        # Mostrar o botão apenas se a arquitetura não for carregada
        if not st.session_state.architecture_loaded:
            # Exibindo a janela de exemplo se o botão for pressionado
            if st.button("Veja exemplo de arquitetura ao final da página", key="toggle_popup"):
                # Alterna entre abrir e fechar o exemplo
                st.session_state.popup_open = not st.session_state.popup_open

        st.write("**Insira os parâmetros do modelo:**")
        img_x = st.number_input("Tamanho da dimensão X (largura):", min_value=1, value=256, step=1)
        img_y = st.number_input("Tamanho da dimensão Y (altura):", min_value=1, value=256, step=1)
        num_classes = st.number_input("Número de classes:", min_value=1, value=2, step=1)

        if uploaded_model and uploaded_architecture:
            # Salvar os arquivos localmente
            with open("uploaded_model.h5", "wb") as model_file:
                model_file.write(uploaded_model.read())

            # Carregar a arquitetura a partir do arquivo .py
            try:
                # Criar o modelo com a função definida na arquitetura
                img_size = (img_y, img_x)  # Ajustar para 3 canais (RGB)
                model = get_model_aurora(img_size, num_classes)

                # Carregar os pesos no modelo
                model.load_weights("uploaded_model.h5")
                st.success("Modelo e arquitetura carregados com sucesso!")

            except Exception as e:
                st.error(f"Erro ao atribuir os pesos à arquitetura, verifique o tamanho da entrada e classes esperadas no output : {e}")
        else:
            if not uploaded_model:
                st.info("Por favor, faça o upload do arquivo de pesos (.h5).")

            if not uploaded_architecture:
                st.info("Por favor, faça o upload do arquivo de arquitetura (.py) que contenha a função `get_model(img_size, num_classes)`.")


# Coluna 2: Inferência
with col2:
    st.header("Predição e Download")

    # Botão para realizar a inferência
    if st.button("Realizar predição"):
        if uploaded_file is not None:
            try:
                model_config = st.session_state.get("model_config", {})

                if model_config["type"] == "pretrained":
                    try:
                        
                        weights_path = get_weights_path(model_config["tumor_type"], model_config["modality"], model_config["model"])

                        if weights_path and os.path.exists(weights_path):
                            if model_config["model"] == 'GTV-MetNet (Takara et al., 2024)':
                                model.load_weights(weights_path)

                            elif model_config["model"] == 'Met-Seg (Ottesen et al., 2020)':
                                # Carregar o peso (mesmo para todos os inputs)
                                model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True)['state_dict'])
                                model = model.to(device)
                                image_modality = model_config["modality"]

                            elif model_config["model"] == 'AURORA (Buchner et al., 2023)':

                                transform = Compose([
                                    EnsureChannelFirst(),  # Garante que o canal seja a primeira dimensão
                                    ScaleIntensity(),      # Normaliza os valores de intensidade
                                    EnsureType()           # Converte para Tensor se necessário
                                ])

                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())  # Escreve o conteúdo do arquivo
                                    temp_path = tmp_file.name  # O caminho do arquivo temporário

                                image_path = temp_path

                                dataset = InferenceDataset(image_path=image_path, transform=transform)
                                data_loader = DataLoader(dataset, batch_size=1, num_workers=0)

                                config = AuroraInfererConfig(
                                    device=Device.GPU if torch.cuda.is_available() else Device.CPU
                                )
                                device = torch.device(config.device.value)
                                model_handler = ModelHandler(config=config, device=device)

                                if model_config['modality'] == "T1 com contraste":
                                    inference_mode = InferenceMode.T1C_O
                                elif model_config['modality'] == "T1 sem contraste":
                                    inference_mode = InferenceMode.T1_O
                                elif model_config['modality'] == "FLAIR":
                                    inference_mode = InferenceMode.FLA_O

                                model_handler.load_aurora_model(inference_mode=inference_mode, num_input_modalities=1)
                                
                                #model = model_handler._load_model(num_input_modalities=1) # Para o Grad-Cam precisa ter a definição do Model

                        else:
                            st.error(f"Pesos não encontrados para {model_config['tumor_type']} na modalidade {model_config['modality']}.")
                            st.stop()
                    except Exception as e:
                        st.error(f"Erro ao carregar pesos: {str(e)}")
                        st.stop()

                elif model_config["type"] == "custom":
                    try:
                        # Carrega o modelo customizado
                        model_path = model_config.get("model_path", "")
                        if model_path and os.path.exists(model_path):
                            model = load_model_or_weights(model_path, model_path, img_size, num_classes)
                        else:
                            st.error("Modelo próprio não encontrado. Verifique o upload.")
                            st.stop()
                    except Exception as e:
                        st.error(f"Erro ao carregar modelo próprio: {str(e)}")
                        st.stop()

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

                # Realizar a segmentação pelo modelo GTV-MetNet
                if model_config["model"] == 'GTV-MetNet (Takara et al., 2024)':

                    progress_bar = st.progress(0)

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

                    st.success("Predição concluída com sucesso!")

                # Realizar a segmentação pelo modelo Met-Seg
                elif model_config["model"] == 'Met-Seg (Ottesen et al., 2020)':

                    with st.spinner("Realizando a predição, por favor aguarde algumas horas..."):
                        model_output, smooth_output = segment_image(tmp_file_path, model, device)[1:]

                    progress_bar = st.progress(0)

                    smooth_segmentations = []
                    segmentations = []
                    num_slices = img_data.shape[2]

                    for i in range(num_slices):
                        slice_image = img_data[:, :, i]  # Fatia 2D
                        mask_class_1 = smooth_output[0, :, :, 1]  # Máscara da classe 1 smooth, o model output é o mesmo que o mask_resized com np.where e threshold
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

                    st.success("Predição concluída com sucesso!")

                elif model_config["model"] == 'AURORA (Buchner et al., 2023)':

                    from time import time
                    
                    start_time = time()

                    with st.spinner("Realizando a predição, por favor aguarde cerca de 2 minutos..."):
                        result = model_handler.infer(data_loader=data_loader)

                    end_time = time()

                    duration = end_time - start_time
                    print(f"O código levou {duration:.4f} segundos para rodar.")

                    progress_bar = st.progress(0)

                    smooth_segmentations = []
                    segmentations = []
                    num_slices = np.shape(result)[2]

                    print(f"result shape: {np.shape(result)} -- num_slices: {num_slices}")

                    for i in range(num_slices):
                        slice_image = img_data[:, :, i]  # Fatia 2D do volume original
                        mask_class_1 = result[:, :, i]  # Máscara correspondente no mesmo slice
                        
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
                    st.success("Predição concluída com sucesso!")
                
                progress_bar.empty()  # Remove a barra de progresso quando concluir

            except Exception as e:
                st.error(f"Erro durante a predição: {str(e)}")
        else:
            st.error("Faça o upload de uma imagem antes de realizar a predição.")

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
            st.warning("Por favor, realize a predição antes de fazer o download.")


# Coluna 3: Visualização
with col3:
    st.header("Visualização")

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
            #image_cam = np.repeat(resized_image[..., np.newaxis], 3, axis=-1)  # Replica o canal para 3 dimensões

            # Gera o Grad-CAM
            #cam_upsample = generate_grad_cam(model, image_cam, 1, 'up_block_4_upsample')
            #cam_residual = generate_grad_cam(model, image_cam, 1, 'up_block_4_residual_conv')

            #if cam_upsample is None or np.array(cam_upsample).ndim == 0 or cam_upsample.size == 0:
            #    cam_upsample = np.zeros((256, 256))  # Matriz de zeros pra substituir o None de quando o Grad-CAM não retornar nada

            #if cam_residual is None or np.array(cam_residual).ndim == 0 or cam_residual.size == 0:
            #    cam_residual = np.zeros((256, 256))  # Matriz de zeros pra substituir o None de quando o Grad-CAM não retornar nada

            #cam_upsample_resized = resize(
            #    cam_upsample,
            #    slice_image.shape,
            #    order=0,
            #    preserve_range=True,
            #    anti_aliasing=True
            #)

            #cam_residual_resized = resize(
            #    cam_residual,
            #    slice_image.shape,
            #    order=0,
            #    preserve_range=True,
            #    anti_aliasing=True
            #)

            image_with_mask = slice_image * smooth_segmentations[slice_idx]
            mask_complement = 1 - smooth_segmentations[slice_idx]
            image_without_mask = slice_image * mask_complement

            # Visualiza a imagem original redimensionada
            st.subheader("Imagem Original")
            fig, ax = plt.subplots()
            ax.imshow(slice_image, cmap='gray')
            ax.set_title(f"Slice {slice_idx} - Imagem Original")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('') 
            ax.set_ylabel('')
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
            ax.set_xticks([])
            ax.set_yticks([]) 
            ax.set_xlabel('')
            ax.set_ylabel('')
            st.pyplot(fig)
    
            # Função para carregar e processar a Ground Truth
            def process_ground_truth(uploaded_file):
                """
                Processa um arquivo NIfTI carregado pelo usuário e retorna os dados da imagem.
                Suporta arquivos .nii e .nii.gz.
                
                Args:
                    uploaded_file: Arquivo enviado via Streamlit (UploadedFile).
                
                Returns:
                    numpy.ndarray: Dados da Ground Truth.
                """
                # Determina o sufixo baseado no nome do arquivo enviado
                suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"

                # Salva o arquivo carregado em um arquivo temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.read())  # Escreve o conteúdo no arquivo
                    temp_file_path = temp_file.name

                # Tenta carregar o arquivo com nibabel
                try:
                    nii_img = nib.load(temp_file_path)
                    ground_truth_data = nii_img.get_fdata()
                    return ground_truth_data

                except nib.filebasedimages.ImageFileError as e:
                    raise ValueError(f"Erro ao carregar o arquivo NIfTI ({uploaded_file.name}): {e}")

            # Inicializa o estado no session_state para persistir o botão pressionado
            if 'ground_truth_loaded' not in st.session_state:
                st.session_state.ground_truth_loaded = False

            if 'uploaded_ground_truth' not in st.session_state:
                st.session_state.uploaded_ground_truth = None

            # Exibe o file_uploader
            uploaded_ground_truth = st.file_uploader("Carregue a Ground Truth em formato NIfTI", type=["nii", "nii.gz"], key="GT")

            # Quando o arquivo for carregado, armazena no session_state
            if uploaded_ground_truth is not None and not st.session_state.ground_truth_loaded:
                st.session_state.uploaded_ground_truth = uploaded_ground_truth
                st.session_state.ground_truth_loaded = True
                st.success(f"Arquivo {uploaded_ground_truth.name} carregado com sucesso!")

            # Exibe o botão "Calcular DSC" e mantém pressionado após a primeira execução
            if st.button("Calcular DSC (Dice Similarity Coefficient)"):

                # Verifica se a Ground Truth já foi carregada
                if st.session_state.uploaded_ground_truth is not None:
                    # Processa a Ground Truth
                    ground_truth_data = process_ground_truth(uploaded_ground_truth)
                    segmentations = st.session_state.segmentations
                    transeg = np.transpose(segmentations, axes = (1,2,0))

                    # Normalização da saída, de forma geral, não é o ideal mas é o mais generalista possível
                    ground_truth_data = np.where(ground_truth_data > 0, 1, 0)

                    slice_dice = calculate_dice_coefficient(ground_truth_data[:, :, slice_idx], transeg[..., slice_idx])
                
                    # Comparação da imagem completa
                    full_dice = calculate_dice_coefficient(ground_truth_data, transeg)

                    # Exibir os resultados
                    st.markdown(f"### Resultados do DSC:")
                    st.write(f"**DSC para o Slice {slice_idx}:** {slice_dice:.4f}")
                    st.write(f"**DSC para a Imagem Completa:** {full_dice:.4f}")
                    
                    # st.write(f"Verificação das dimensões da Ground Truth: {ground_truth_data.shape}") # Usado para debuggar

                else:
                    st.warning("Por favor, carregue a Ground Truth antes de calcular o DSC.")

                plot_matrix = np.zeros_like(ground_truth_data[..., slice_idx], dtype=np.uint8)

                from matplotlib.colors import ListedColormap
                from matplotlib.patches import Patch

                # Definir condições
                plot_matrix[(ground_truth_data[..., slice_idx] == 1) & (transeg[..., slice_idx] == 0)] = 1  # Vermelho
                plot_matrix[(ground_truth_data[..., slice_idx] == 0) & (transeg[..., slice_idx] == 1)] = 2  # Cinza
                plot_matrix[(ground_truth_data[..., slice_idx] == 1) & (transeg[..., slice_idx] == 1)] = 3  # Verde
                cmap = ListedColormap(["black", "red", "gray", "green"])

                fig, ax = plt.subplots()
                ax.imshow(plot_matrix, cmap=cmap, alpha=1.0)  # Aplicar o colormap
                ax.set_title(f"Slice {slice_idx} - Diferenças entre GT e Predição")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")


                # Criar patches para a legenda
                legend_patches = [
                    Patch(color="red", label="Tumor não encontrado (FN)"),
                    Patch(color="gray", label="Detecção errônea (FP)"),
                    Patch(color="green", label="Tumor encontrado (TP)"),
                ]

                ax.legend(handles=legend_patches, loc="best", fontsize='small')
                        
                st.pyplot(fig)

        else:
            st.warning("Por favor, realize a predição antes de visualizar os slices.")

    else:
        st.warning("Faça o upload de uma imagem para começar.")

if model_choice == "Modelo Próprio":
    # Se a janela de exemplo estiver aberta
        if st.session_state.popup_open:
            # Exibe o código de exemplo
            code = """
    from tensorflow.keras import layers, Model, Input  # Import necessário
    from tensorflow import keras  # Import necessário para `keras.Input`

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
            """
            st.code(code, language="python")

            # Botão de fechar exemplo
            if st.button("Fechar Exemplo", key="close_popup"):
                st.session_state.popup_open = False  # Fecha o pop-up