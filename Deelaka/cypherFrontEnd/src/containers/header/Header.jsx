import React, { useEffect, useRef } from 'react';
import './header.css';
import axios from 'axios';
import io from 'socket.io-client';

const Header = () => {
  const videoRef = useRef(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      })
      .catch(error => {
        console.error('Error accessing camera:', error);
      });
  }, []);

  const handleDropdownChange = (event) => {
    const selectedValue = event.target.value;
    if (selectedValue === 'vegetable') {
      // start intrusion detection
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      const frameRate = 30;
      setInterval(() => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg');
        axios.post('http://localhost:8000/intrusion-detection', { imageData })
          .then(response => {
            // handle response
            console.log('Intrusion detection response:', response.data);
            if (response.data.intrusionDetected) {
              // show an alert for intrusion detection
              alert('Intrusion detected!');
            }
          })
          .catch(error => {
            // handle error
            console.error('Error sending image data for intrusion detection:', error);
          });
      }, 1000 / frameRate);

      const socket = io('http://localhost:8000');
      socket.on('intrusion-detected', data => {
        // handle the intrusion detection alert
        alert('Intrusion Detected!');
      });
    }
  };

  return (
    <div className='screen'>
      <select className='dropdown' onChange={handleDropdownChange}>
        <option value="violence">Violence Detection</option>
        <option value="intrusion">Intrusion Detection</option>
        <option value="reid">Person re-identification</option>
      </select>
      <video ref={videoRef} className='video' />
    </div>
  );
} 

export default Header;

