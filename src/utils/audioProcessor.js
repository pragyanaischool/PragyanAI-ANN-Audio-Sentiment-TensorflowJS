// src/utils/audioProcessor.js
import Meyda from 'meyda';

const SAMPLE_RATE = 22050; // Matches your model training rate
const N_MFCC = 40;           // Matches your model input shape [-1, 40][cite: 2]
const HOP_LENGTH = 512;
const N_FFT = 2048;
const DURATION_SEC = 3.0;    // Target duration for analysis

const EXPECTED_SAMPLES = SAMPLE_RATE * DURATION_SEC;

/**
 * Processes an AudioBuffer to extract Mean MFCC features with NaN safety fallback.
 */
export const processAudioForPrediction = async (audioBuffer) => {
    const offlineContext = new OfflineAudioContext(
        1, // Mono channel
        EXPECTED_SAMPLES,
        SAMPLE_RATE
    );

    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start(0);

    const renderedBuffer = await offlineContext.startRendering();
    const channelData = renderedBuffer.getChannelData(0);

    const frameSize = N_FFT;
    const hop = HOP_LENGTH;
    const numFrames = Math.floor((channelData.length - frameSize) / hop) + 1;
    
    let mfccSum = new Array(N_MFCC).fill(0);
    let framesProcessed = 0;

    for (let i = 0; i < numFrames; i++) {
        const start = i * hop;
        const frame = channelData.slice(start, start + frameSize);
        
        if (frame.length === frameSize) {
             const features = Meyda.extract('mfcc', frame, {
                 n_mfcc: N_MFCC,
                 n_fft: N_FFT,
             });
             
             if (features) {
                let frameValid = true;
                for (let j = 0; j < N_MFCC; j++) {
                    if (isNaN(features[j]) || !isFinite(features[j])) {
                        frameValid = false;
                        break;
                    }
                }

                // Only add frame if all coefficients are valid numbers
                if (frameValid) {
                    for (let j = 0; j < N_MFCC; j++) {
                        mfccSum[j] += features[j];
                    }
                    framesProcessed++;
                }
             }
        }
    }

    // If no valid frames were found (e.g., total silence), return a clean zero-filled array
    if (framesProcessed === 0) {
        console.warn("Warning: Audio signal was silent or yielded no valid frames. Returning zero-tensor.");
        return new Float32Array(N_MFCC).fill(0);
    }

    // Average the features across all valid frames (equivalent to librosa .mean(axis=1))
    const meanMfccs = mfccSum.map(sum => sum / framesProcessed);
    
    return new Float32Array(meanMfccs);
};
