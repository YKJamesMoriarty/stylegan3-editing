import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import dlib
import subprocess

from utils.alignment_utils import align_face, crop_face, get_stylegan_transform


ENCODER_PATHS = {
    "restyle_e4e_ffhq": {"id": "1z_cB187QOc6aqVBdLvYvBjoc93-_EuRm", "name": "restyle_e4e_ffhq.pt"},
    "restyle_pSp_ffhq": {"id": "12WZi2a9ORVg-j6d9x4eF-CKpLaURC2W-", "name": "restyle_pSp_ffhq.pt"},
}
INTERFACEGAN_PATHS = {
    "age": {'id': '1NQVOpKX6YZKVbz99sg94HiziLXHMUbFS', 'name': 'age_boundary.npy'},
    "smile": {'id': '1KgfJleIjrKDgdBTN4vAz0XlgSaa9I99R', 'name': 'Smiling_boundary.npy'},
    "pose": {'id': '1nCzCR17uaMFhAjcg6kFyKnCCxAKOCT2d', 'name': 'pose_boundary.npy'},
    "Male": {'id': '18dpXS5j1h54Y3ah5HaUpT03y58Ze2YEY', 'name': 'Male_boundary.npy'}
}
STYLECLIP_PATHS = {
    "delta_i_c": {"id": "1HOUGvtumLFwjbwOZrTbIloAwBBzs2NBN", "name": "delta_i_c.npy"},
    "s_stats": {"id": "1FVm_Eh7qmlykpnSBN1Iy533e_A2xM78z", "name": "s_stats"},
}


class Downloader:

    def __init__(self, code_dir, use_pydrive, subdir):
        self.use_pydrive = use_pydrive
        current_directory = os.getcwd()
        self.save_dir = os.path.join(os.path.dirname(current_directory), code_dir, subdir)
        os.makedirs(self.save_dir, exist_ok=True)
        if self.use_pydrive:
            self.authenticate()

    def authenticate(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def download_file(self, file_id, file_name):
        file_dst = f'{self.save_dir}/{file_name}'
        if os.path.exists(file_dst):
            print(f'{file_name} already exists!')
            return
        if self.use_pydrive:
            downloaded = self.drive.CreateFile({'id': file_id})
            downloaded.FetchMetadata(fetch_all=True)
            downloaded.GetContentFile(file_dst)
        else:
            command = self._get_download_model_command(file_id=file_id, file_name=file_name)
            subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    def _get_download_model_command(self, file_id, file_name):
        """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
        url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=self.save_dir)
        return url


def download_dlib_models():
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')


def run_alignment(image_path):
    #确保使用 dlib 的面部标志检测模型（shape_predictor_68_face_landmarks.dat）已经下载并准备好
    download_dlib_models()
    #predictor：任务是找到图像中人脸的大致位置，会返回一个矩形框
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #detector：在 detector 检测到的人脸区域中，进一步精确预测面部关键点的位置，总共有 68 个标准点
    detector = dlib.get_frontal_face_detector()
    print("Aligning image...")
    #使用上述的面部检测器（detector）和面部标志预测器（predictor）对图像中的面部进行对齐，确保面部朝向标准化。
    aligned_image = align_face(filepath=str(image_path), detector=detector, predictor=predictor)
    print(f"Finished aligning image: {image_path}")
    return aligned_image

'''
#作用：对输入的图像进行面部裁剪，使用 dlib 库中的面部检测器和面部关键点预测模型
'''
def crop_image(image_path):
    download_dlib_models()#这个函数确保面部检测所需的模型文件已被下载到本地。如果模型未下载，函数会下载必要的文件
    '''这个语句加载了基于 dlib 的面部关键点预测器。shape_predictor 通过读取 .dat 文件，
    可以在图像中标注出 68 个面部关键点，用于进一步的面部区域处理。'''
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    '''get_frontal_face_detector是一个预训练的正面人脸检测器，dlib 提供了这个工具，用于
    识别图像中的正面人脸。它会返回一组检测到的面部矩形区域，用于进一步处理'''
    detector = dlib.get_frontal_face_detector()
    print("Cropping image...")
    '''crop_face 是实际执行面部裁剪的函数，它将图像路径、面部检测器和面部标志预测器作为输入。
    --crop_face 是实际执行面部裁剪的函数，它将图像路径、面部检测器和面部标志预测器作为输入。
    --它首先使用 detector 来检测图像中的面部区域。
    --然后，使用 predictor 来定位面部中的关键点（如眼睛、鼻子等），确定精确的面部轮廓。
    --根据这些信息，裁剪出图像中的面部部分，并返回裁剪后的图像'''
    cropped_image = crop_face(filepath=str(image_path), detector=detector, predictor=predictor)
    print(f"Finished cropping image: {image_path}")
    return cropped_image#返回裁剪后的图像

'''
compute_transforms函数的主要目的是通过检测面部关键点，计算从裁剪后的面部图像到对齐后的面部图像之间的转换矩阵'''
def compute_transforms(aligned_path, cropped_path):
    download_dlib_models()
    #加载 dlib 提供的 68 个面部关键点检测器模型。这个模型会在面部图像中定位眼睛、鼻子、嘴巴等关键点
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #初始化正面人脸检测器，它用于检测输入图像中是否存在人脸，并返回面部的矩形区域。
    detector = dlib.get_frontal_face_detector()
    print("Computing landmarks-based transforms...")
    '''get_stylegan_transform 函数是一个核心函数，它处理两个输入图像，计算它们之间的转换矩阵。 
    参数：
    cropped_path: 裁剪后图像的路径。
    aligned_path: 对齐后图像的路径。
    detector: 人脸检测器，用于定位人脸区域。
    predictor: 面部关键点预测器，用于检测面部关键点。
    get_stylegan_transform 函数会比较裁剪图像与对齐图像的关键点位置，通过这些位置计算出图像的变换参数，
    比如旋转角度、平移量、以及变换矩阵。这些变换能够将裁剪后的面部与对齐后的面部图像对齐。
    '''
    res = get_stylegan_transform(str(cropped_path), str(aligned_path), detector, predictor)
    print("Done!")
    if res is None:
        print(f"Failed computing transforms on: {cropped_path}")
        return
    else:
        '''res 是 get_stylegan_transform 返回的结果，包含四个部分：
        rotation_angle: 图像的旋转角度。
        translation: 图像的平移参数。
        transform: 图像的变换矩阵。
        inverse_transform: 图像的逆变换矩阵。'''
        rotation_angle, translation, transform, inverse_transform = res
        return inverse_transform
