from scipy.io import loadmat

sample_file = "wifi_csi/mat/act_1_1.mat"

try:
    mat_data = loadmat(sample_file)
    trace_data = mat_data['trace']
    
    # เจาะเข้าไปดูข้อมูลใน Packet แรกสุด (กล่องใบแรก)
    first_packet = trace_data[0, 0]
    
    print("เจาะกล่องสำเร็จ!")
    
    # เช็คว่าข้างในมีฟิลด์อะไรซ่อนอยู่บ้าง
    if hasattr(first_packet, 'dtype') and first_packet.dtype.names is not None:
        print("ตัวแปรที่ซ่อนอยู่ใน Packet คือ:", first_packet.dtype.names)
    else:
        # ถ้าไม่มีชื่อฟิลด์ ลองปริ้นท์รูปร่างมันออกมา
        print("รูปร่างข้อมูลข้างใน:", first_packet.shape)
        
except Exception as e:
    print("เกิดข้อผิดพลาด:", e)