import React, { useState } from 'react';
import axios from 'axios';

function ImageUpload() {
    const [images, setImages] = useState([]);
    const [boundingBoxFile, setBoundingBoxFile] = useState(null);  // Add state for bounding box file

    const handleFileChange = (e) => {
        setImages(e.target.files);
    };

    const handleBoundingBoxFileChange = (e) => {
        setBoundingBoxFile(e.target.files[0]);  // Handle single bounding box file
    };

    const handleSubmit = async () => {
        const formData = new FormData();
        for (let i = 0; i < images.length; i++) {
            formData.append('images', images[i]);
        }

        if (boundingBoxFile) {
            formData.append('bounding_boxes', boundingBoxFile);  // Append bounding box file
        }
    
        try {
            const response = await axios.post('http://localhost:5000/upload_images', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
    
            console.log('Response:', response.data);
        } catch (error) {
            console.error('Error uploading images:', error);
        }
    };

    return (
        <div>
            <h2>Upload Image Pairs</h2>
            <input type="file" multiple onChange={handleFileChange} />
            <h2>Upload Bounding Box File (Optional)</h2>
            <input type="file" onChange={handleBoundingBoxFileChange} />
            <button onClick={handleSubmit}>Upload and Process</button>
        </div>
    );
}

export default ImageUpload;
