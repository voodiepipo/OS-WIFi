import torch
from torch.utils.data import DataLoader
from Model import DualStreamModel
from Utils import CSIDataset

if __name__ == '__main__':
    # -------- Load Data --------
    # ชี้เป้าไปที่โฟลเดอร์ไฟล์ข้อมูลดิบที่คุณโหลดมา
    csv_path = "annotation.csv"
    mat_path = "wifi_csi/mat"
    
    print("กำลังเตรียมชุดข้อมูล...")
    train_dataset = CSIDataset(csv_path, mat_path)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # -------- Device Setup (สำคัญสำหรับ Mac M3) --------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"เริ่มรันเทรนโมเดลบน: {device}")
    
    # -------- Model Setup --------
    model = DualStreamModel(num_classes=9).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # -------- Training Loop --------
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # วนลูปดึงข้อมูลทีละ 8 ไฟล์ (Lazy Loading)
        for batch_idx, (amp, phase, label) in enumerate(train_loader):
            amp = amp.to(device)
            phase = phase.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            outputs = model(amp, phase)
            
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # ปริ้นท์บอกความคืบหน้าทุกๆ 100 batch (จะได้รู้ว่าไม่ค้าง)
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            # เติม 2 บรรทัดนี้ไว้ล่างสุดของไฟล์
           
        torch.save(model.state_dict(), "wifi_model.pth")
        print("🎉 เซฟโมเดลสำเร็จ! บันทึกเป็นไฟล์ wifi_model.pth")
        avg_loss = total_loss / len(train_loader)
        print(f"=== จบ Epoch {epoch+1}, Average Loss: {avg_loss:.4f} ===")