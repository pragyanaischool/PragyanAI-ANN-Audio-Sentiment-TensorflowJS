import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { processAudioForPrediction } from './utils/audioProcessor';
import './App.css';

const MODEL_URL = `${process.env.PUBLIC_URL}/model/model.json`;

// Emotion mapping corresponding to your 8 output classes[cite: 2]
const emotionMap = {
  0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad", 
  4: "Angry", 5: "Fearful", 6: "Disgusted", 7: "Surprised"
};

function App() {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('Initializing Model...');
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);

  // 1. Load TF.js Graph Model on Startup
  useEffect(() => {
    const loadModel = async () => {
      try {
        setStatus('Loading TensorFlow.js Graph Model...');
        const loadedModel = await tf.loadGraphModel(MODEL_URL);
        setModel(loadedModel);
        setLoading(false);
        setStatus('Model Loaded Successfully. Ready.');
        
        // Warm up the model with a dummy tensor [1, 40][cite: 2]
        const dummyInput = tf.zeros([1, 40]);
        loadedModel.predict(dummyInput);
        dummyInput.dispose();
      } catch (error) {
        console.error('Model loading error:', error);
        setStatus('Failed to load model. Check console.');
      }
    };
    loadModel();
  }, []);

  // 2. Run Inference Pipeline
  const runInference = async (audioBuffer) => {
    if (!model) return;
    setLoading(true);
    setPrediction(null);
    setStatus('Extracting audio features...');

    try {
      const mfccFeatures = await processAudioForPrediction(audioBuffer);
      
      // Create input tensor matching shape [1, 40][cite: 2]
      const inputTensor = tf.tensor2d(mfccFeatures, [1, mfccFeatures.length]);

      setStatus('Running sentiment analysis...');
      const outputTensor = model.predict(inputTensor);
      const probabilities = await outputTensor.data();

      const predictedClass = Array.from(probabilities).indexOf(Math.max(...probabilities));
      const confidence = probabilities[predictedClass];
      
      setPrediction({
        emotion: emotionMap[predictedClass] || `Class ${predictedClass}`,
        confidence: (confidence * 100).toFixed(2)
      });
      
      setStatus('Analysis Complete.');

      inputTensor.dispose();
      outputTensor.dispose();
    } catch (error) {
      console.error('Inference error:', error);
      setStatus('Error during inference execution.');
    } finally {
      setLoading(false);
    }
  };

  // 3. Microphone Recording Control
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioUrl(URL.createObjectURL(audioBlob));

        if (!audioContextRef.current) {
           audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        }

        const arrayBuffer = await audioBlob.arrayBuffer();
        const decodedAudio = await audioContextRef.current.decodeAudioData(arrayBuffer);
        runInference(decodedAudio);
      };

      mediaRecorderRef.current.start();
      setRecording(true);
      setStatus('Recording live audio...');
    } catch (err) {
      console.error("Microphone permission denied:", err);
      setStatus("Microphone access denied.");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  // 4. File Upload Handler
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }

    fetch(url)
      .then(res => res.arrayBuffer())
      .then(buf => audioContextRef.current.decodeAudioData(buf))
      .then(decoded => runInference(decoded));
  };

  return (
    <div className="App" style={{ textAlign: 'center', padding: '40px' }}>
      <h1>🎙️ Client-Side Audio Sentiment Analysis</h1>
      <p>Powered by React & TensorFlow.js</p>
      
      <div style={{ margin: '20px 0', padding: '10px', background: '#f4f4f4', display: 'inline-block', borderRadius: '5px' }}>
        Status: <strong>{status}</strong>
      </div>

      <div style={{ margin: '20px 0' }}>
        <button onClick={startRecording} disabled={recording || loading || !model} style={{ padding: '10px 20px', marginRight: '10px' }}>
          {recording ? '🔴 Recording...' : '🎤 Record Audio'}
        </button>
        <button onClick={stopRecording} disabled={!recording} style={{ padding: '10px 20px' }}>
          ⏹️ Stop
        </button>
      </div>

      <div style={{ margin: '20px 0' }}>
        <label>Or upload audio file: </label>
        <input type="file" accept="audio/*" onChange={handleFileUpload} disabled={loading || !model} />
      </div>

      {audioUrl && (
        <div style={{ margin: '20px 0' }}>
          <audio controls src={audioUrl}></audio>
        </div>
      )}

      {prediction && (
        <div style={{ marginTop: '20px', padding: '20px', border: '1px solid #ccc', display: 'inline-block', borderRadius: '8px', background: '#fff' }}>
          <h2>Result</h2>
          <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#2c3e50' }}>{prediction.emotion}</p>
          <p>Confidence: {prediction.confidence}%</p>
        </div>
      )}
    </div>
  );
}

export default App;
