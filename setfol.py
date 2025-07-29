import os

def create_project_structure():
    """Buat struktur folder untuk face recognition project"""
    
    # Daftar folder yang perlu dibuat
    folders = [
        'data',
        'data/known_faces',
        'data/encodings',
        'src',
        'models'
    ]
    
    print("🚀 Membuat struktur folder project...")
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Created: {folder}")
        else:
            print(f"📁 Already exists: {folder}")
    
    # Buat file __init__.py di folder src
    init_file = 'src/__init__.py'
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Face Recognition System\n')
        print(f"✅ Created: {init_file}")
    
    # Buat contoh folder untuk 3 person
    example_persons = ['john_doe', 'jane_smith', 'alex_johnson']
    
    for person in example_persons:
        person_folder = f'data/known_faces/{person}'
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
            print(f"✅ Created example folder: {person_folder}")
    
    print("\n🎉 Struktur folder berhasil dibuat!")
    print("\n📋 Langkah selanjutnya:")
    print("   1. Copy foto training ke folder data/known_faces/[nama_person]/")
    print("   2. Setiap person minimal 5-10 foto")
    print("   3. Format foto: JPG, PNG, atau JPEG")
    print("   4. Pastikan hanya ada 1 wajah per foto")

if __name__ == "__main__":
    create_project_structure()