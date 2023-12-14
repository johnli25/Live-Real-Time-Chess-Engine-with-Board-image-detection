heic_files = os.listdir(heic_directory)
def heic_to_jpg(heic_directory, output_directory):
    for filename in heic_files:
        print("directory + filename:", os.path.join(heic_directory, filename))
        heif_file = pillow_heif.read_heif(os.path.join(heic_directory, filename))
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        image.save(os.path.join(output_directory, filename.replace(".HEIC", ".jpg")), format="jpeg")


