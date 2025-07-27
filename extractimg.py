import fitz  # PyMuPDF
import uuid

def extract_image_locations(pdf_path):
    doc = fitz.open(pdf_path)
    image_tags = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            bbox = img[1]  # Bounding box
            unique_id = str(uuid.uuid4())
            
            image_tags.append({
                "id": unique_id,
                "page": page_num + 1,
                "bbox": bbox,
                "image_data": image_bytes
            })
    
    return image_tags

# Example usage
pdf_path = r"C:\VIT\6th semester\Game Programming\Lab\Lab1_22BAI1377.pdf"
images = extract_image_locations(pdf_path)
print(images)