from ultralytics import YOLO

if __name__ == "__main__":
    
    model = YOLO("yolov8n.pt")  

    
    model.train(
        data="config.yaml",   
        epochs=100,
        batch=16,             
        imgsz=640,
        device=0,             
        optimizer="AdamW",
        amp=True,             
        name="treino_rachaduras"
    )
