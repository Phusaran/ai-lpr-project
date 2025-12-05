import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [logs, setLogs] = useState([]);
  const [videoUrl, setVideoUrl] = useState('http://localhost:5001/video_feed');
  const [isPaused, setIsPaused] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(() => {
      fetch('http://localhost:5001/get_logs')
        .then(res => res.json())
        .then(data => setLogs(data))
        .catch(err => console.error("Error:", err));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      await fetch('http://localhost:5001/upload_video', { method: 'POST', body: formData });
      setVideoUrl(`http://localhost:5001/video_feed?t=${Date.now()}`);
      setIsPaused(false);
    } catch (error) {
      alert("Upload failed!");
    }
  };

  const handleTogglePlayback = async () => {
    try {
      const res = await fetch('http://localhost:5001/toggle_playback', { method: 'POST' });
      const data = await res.json();
      setIsPaused(data.is_paused);
    } catch (error) { console.error(error); }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>🚗 AI License Plate Recognition</h1>
        <div className="controls">
          <input 
            type="file" accept="video/*" style={{ display: 'none' }} 
            ref={fileInputRef} onChange={handleFileUpload}
          />
          <button className="btn btn-upload" onClick={() => fileInputRef.current.click()}>
            📂 Import Video
          </button>
          <button className={`btn ${isPaused ? 'btn-play' : 'btn-pause'}`} onClick={handleTogglePlayback}>
            {isPaused ? "▶️ Play" : "⏸️ Pause"}
          </button>
        </div>
      </header>

      <div className="content">
        <div className="video-section">
          <div className="video-container">
            <img 
              key={videoUrl} src={videoUrl} alt="LPR Stream" className="video-feed" 
              onError={(e) => {
                e.target.onerror = null; 
                e.target.src = "https://via.placeholder.com/640x480?text=Waiting+for+Backend...";
              }}
            />
          </div>
          <div className="status-bar">
            <span className="dot" style={{backgroundColor: isPaused ? 'orange' : '#0f0'}}></span> 
            {isPaused ? "Paused" : "System Active (Port 5001)"}
          </div>
        </div>

        <div className="log-section">
          <h2>📋 Detection Log</h2>
          <div className="log-list">
            <table>
              <thead>
                <tr><th>Time</th><th>License Plate</th></tr>
              </thead>
              <tbody>
                {logs.length === 0 ? (
                  <tr><td colSpan="2" style={{textAlign: 'center'}}>No detections yet...</td></tr>
                ) : (
                  logs.map((log, index) => (
                    <tr key={index} className={index === 0 ? "new-log" : ""}>
                      <td>{log.time}</td><td>{log.text}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App