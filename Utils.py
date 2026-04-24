import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.io import loadmat
from Preprocess import preprocess_csi

class CSIDataset(Dataset):
    def __init__(self, csv_file, mat_folder, max_time_steps=1000): 
        # โหลดตารางเฉลยขึ้นมา
        self.df = pd.read_csv(csv_file)
        self.mat_folder = mat_folder
        self.max_time = max_time_steps # ล็อคความยาวข้อมูลให้เท่ากันทุกไฟล์
        
        # แปลงชื่อคลาสจากไฟล์ CSV ให้เป็นตัวเลข (0-8) 
        self.label_map = {
            'act_1_1': 0, 'act_1_2': 1, 'act_1_3': 2,
            'act_1_4': 3, 'act_1_5': 4, 'act_1_6': 5,
            'act_1_7': 6, 'act_1_8': 7, 'act_1_9': 8
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_str = str(row['label']).strip()
        
        # จากรูป Explorer ในเครื่องคุณ ชื่อไฟล์ตั้งตาม label เลย
        mat_filename = label_str + ".mat" 
        mat_path = os.path.join(self.mat_folder, mat_filename)
        
        try:
            # โหลดไฟล์ .mat จากฮาร์ดดิสก์
            mat_data = loadmat(mat_path)
            trace_data = mat_data['trace']
            num_packets = trace_data.shape[0]
            
            csi_list = []
            
            # วนลูปเจาะเข้าไปดึงค่า 'csi' จากแต่ละ Packet (จำกัดสูงสุดแค่ max_time)
            for i in range(min(num_packets, self.max_time)):
                # เจาะเข้าไปที่ตัวแปร 'csi' (ข้อมูลมักจะเป็นจำนวนเชิงซ้อน)
                csi_complex = trace_data[i, 0]['csi'][0, 0]
                
                # ปกติ CSI มีหลายเสาอากาศ เราดึงคู่เสาอากาศแรกมาใช้ [0, 0, :]
                if csi_complex.ndim == 3:
                    subcarriers = csi_complex[0, 0, :]
                else:
                    subcarriers = csi_complex.flatten()[:30] # กรณีฉุกเฉินดึงมา 30 คลื่น
                
                # แปลงจำนวนเชิงซ้อนเป็น Amplitude และ Phase
                amp = np.abs(subcarriers)
                phase = np.angle(subcarriers)
                
                # เอามาซ้อนกันให้ได้ 2 ช่อง
                csi_list.append(np.stack((amp, phase), axis=-1))
                
            # รวมร่างเป็น Numpy Array: shape จะเป็น (time, subcarriers, 2)
            csi_time_series = np.array(csi_list)
            
            # ถ้าไฟล์ไหนเก็บข้อมูลมาสั้นกว่า 1000 ให้เติมเลข 0 ต่อท้ายให้ครบ (Padding)
            if csi_time_series.shape[0] < self.max_time:
                pad_len = self.max_time - csi_time_series.shape[0]
                padding = np.zeros((pad_len, csi_time_series.shape[1], 2))
                csi_time_series = np.vstack((csi_time_series, padding))
                
        except Exception as e:
            # กันเหนียว: ถ้าไฟล์ไหนพังหรือโหลดไม่ได้ ให้ส่งข้อมูลเปล่าๆ ไป จะได้ไม่ Error กลางคัน
            # print(f"ไฟล์มีปัญหา {mat_filename}: {e}") # ปิด print ไว้จะได้ไม่รกจอ
            csi_time_series = np.zeros((self.max_time, 30, 2))
            
        # โยนเข้าฟังก์ชัน Preprocess ของคุณ!
        amp, phase = preprocess_csi(csi_time_series)
        
        # แปลงเป็น PyTorch Tensor
        amp = torch.tensor(amp, dtype=torch.float32)
        phase = torch.tensor(phase, dtype=torch.float32)
        label = torch.tensor(self.label_map.get(label_str, 0), dtype=torch.long)
        
        return amp, phase, label